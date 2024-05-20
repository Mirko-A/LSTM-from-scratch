import numpy as np
import tqdm
import json

dataset_path = "datasets/shakespeare/tiny_shakespeare_small.txt"
with open(dataset_path, 'r') as file:
    data = file.read()

#? NOTE: Mirko
# Need to use dict.fromKeys to remove duplicates
# since set() does not maintain the order of its
# elements on each run.
vocab = list(dict.fromkeys(data))
vocab_size = len(vocab)

char_to_idx = {c:i for i, c in enumerate(vocab)}
idx_to_char = {i:c for i, c in enumerate(vocab)}

data_size = len(data)
X_train = data[:-1]
Y_train = data[1:]

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)
def dtanh(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x)**2

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))
def dsigmoid(x: np.ndarray) -> np.ndarray:
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x)
    return exp/np.sum(exp)

def one_hot_encode(c: str) -> np.ndarray:
    one_hot = np.zeros((vocab_size, 1))
    one_hot[char_to_idx[c]] = 1
    
    return one_hot

def init_weights(input_size: int, output_size: int) -> np.ndarray:
    # Normalized Xavier initialization
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6/(input_size + output_size))

class LSTM:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 1e-3):
        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Forget gate weights
        self.W_f = init_weights(input_size, self.hidden_size)
        self.b_f = np.zeros((self.hidden_size, 1))

        # Input gate weights
        self.W_i = init_weights(input_size, self.hidden_size)
        self.b_i = np.zeros((self.hidden_size, 1))

        # Candidate gate weights
        self.W_c = init_weights(input_size, self.hidden_size)
        self.b_c = np.zeros((self.hidden_size, 1))

        # Output gate weights
        self.W_o = init_weights(input_size, self.hidden_size)
        self.b_o = np.zeros((self.hidden_size, 1))

        # Final gate weights
        self.W_y = init_weights(self.hidden_size, output_size)
        self.b_y = np.zeros((output_size, 1))
        
        # Network cache
        self.concat_inputs: dict[int, np.ndarray] = {}

        self.hidden_states: dict[int, np.ndarray] = {-1: np.zeros((self.hidden_size, 1))}
        self.cell_states: dict[int, np.ndarray] = {-1: np.zeros((self.hidden_size, 1))}

        self.forget_gates: dict[int, np.ndarray] = {}
        self.input_gates: dict[int, np.ndarray] = {}
        self.candidate_gates: dict[int, np.ndarray] = {}
        self.output_gates: dict[int, np.ndarray] = {}
        self.activation_outputs: dict[int, np.ndarray] = {}
        self.outputs: dict[int, np.ndarray] = {}
        
    def reset_cache(self):
        self.concat_inputs = {}
        
        self.hidden_states = {-1: np.zeros((self.hidden_size, 1))}
        self.cell_states = {-1: np.zeros((self.hidden_size, 1))}
        
        self.forget_gates = {}
        self.input_gates = {}
        self.candidate_gates = {}
        self.output_gates = {}
        self.activation_outputs = {}
        self.outputs = {}
        
    def forward(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        self.reset_cache()
        
        outputs = []
        for t in range(len(inputs)):
            self.concat_inputs[t] = np.concatenate((self.hidden_states[t-1], inputs[t]))
            
            self.forget_gates[t] = np.dot(self.W_f, self.concat_inputs[t]) + self.b_f
            self.input_gates[t] = np.dot(self.W_i, self.concat_inputs[t]) + self.b_i
            self.candidate_gates[t] = np.dot(self.W_c, self.concat_inputs[t]) + self.b_c
            self.output_gates[t] = np.dot(self.W_o, self.concat_inputs[t]) + self.b_o
            
            fga = sigmoid(self.forget_gates[t])
            iga = sigmoid(self.input_gates[t])
            cga = tanh(self.candidate_gates[t])
            oga = sigmoid(self.output_gates[t])
            
            self.cell_states[t] = fga * self.cell_states[t-1] + iga * cga
            self.hidden_states[t] = oga * tanh(self.cell_states[t])
            
            outputs += [np.dot(self.W_y, self.hidden_states[t]) + self.b_y]
            
        return outputs
    
    def backward(self, errors: list[np.ndarray]) -> None:
        dW_f, db_f = 0, 0
        dW_i, db_i = 0, 0
        dW_c, db_c = 0, 0
        dW_o, db_o = 0, 0
        dW_y, db_y = 0, 0
        
        hidden_state = np.zeros_like(self.hidden_states[0])
        cell_state = np.zeros_like(self.cell_states[0])
        
        for t in reversed(range(len(self.concat_inputs))):
            err = errors[t]
            dW_y += np.dot(err, self.hidden_states[t].T)
            db_y += err
            
            dhidden = np.dot(self.W_y.T, err) + hidden_state
            doutput = tanh(self.cell_states[t]) * dhidden * dsigmoid(self.output_gates[t])
            
            dW_o += np.dot(doutput, self.concat_inputs[t].T)
            db_o += doutput
            
            dcell_state = dtanh(self.cell_states[t]) * sigmoid(self.output_gates[t]) * dhidden + cell_state
            
            dforget = dcell_state * self.cell_states[t-1] * dsigmoid(self.forget_gates[t])
            
            dW_f += np.dot(dforget, self.concat_inputs[t].T)
            db_f += dforget
            
            dinput = dcell_state * tanh(self.candidate_gates[t]) * dsigmoid(self.input_gates[t])
            
            dW_i += np.dot(dinput, self.concat_inputs[t].T)
            db_i += dinput
            
            dcandidate = dcell_state * sigmoid(self.input_gates[t]) * dtanh(self.candidate_gates[t])
            
            dW_c += np.dot(dcandidate, self.concat_inputs[t].T)
            db_c += dcandidate
            
            d_concat_inputs = np.dot(self.W_f.T, dforget) + np.dot(self.W_i.T, dinput) + np.dot(self.W_c.T, dcandidate) + np.dot(self.W_o.T, doutput)
            
            hidden_state = d_concat_inputs[:self.hidden_size, :]
            cell_state = sigmoid(self.forget_gates[t]) * dcell_state
            
        for _d in (dW_f, db_f, dW_i, db_i, dW_c, db_c, dW_o, db_o, dW_y, db_y):
            np.clip(_d, -1, 1, out=_d)
            
        self.W_f -= dW_f * self.learning_rate
        self.b_f -= db_f * self.learning_rate
        
        self.W_i -= dW_i * self.learning_rate
        self.b_i -= db_i * self.learning_rate
        
        self.W_c -= dW_c * self.learning_rate
        self.b_c -= db_c * self.learning_rate
        
        self.W_o -= dW_o * self.learning_rate
        self.b_o -= db_o * self.learning_rate
        
        self.W_y -= dW_y * self.learning_rate
        self.b_y -= db_y * self.learning_rate
        
    def train(self, inputs: list[str], labels: list[str], epochs: int = 1):
        one_hot_inputs = [one_hot_encode(c) for c in inputs]
        
        for _ in tqdm.tqdm(range(epochs)):
            predictions = self.forward(one_hot_inputs)
                
            errors = []
            for i in range(len(predictions)):
                errors += [softmax(predictions[i])]
                errors[-1][char_to_idx[labels[i]]] -= 1

            self.backward(errors)
    
    def save_weights(self, weights_path: str) -> None:
        weights = {
            'W_f': self.W_f.tolist(),
            'b_f': self.b_f.tolist(),
            'W_i': self.W_i.tolist(),
            'b_i': self.b_i.tolist(),
            'W_c': self.W_c.tolist(),
            'b_c': self.b_c.tolist(),
            'W_o': self.W_o.tolist(),
            'b_o': self.b_o.tolist(),
            'W_y': self.W_y.tolist(),
            'b_y': self.b_y.tolist(),
        }
            
        
        with open(weights_path, 'w') as f:
            json.dump(weights, f)
    
    def load_weights(self, weights_path: str) -> None:
        with open(weights_path, 'r') as f:
            weights = json.load(f)
            
            self.W_f = np.array(weights['W_f'])
            self.b_f = np.array(weights['b_f'])
            
            self.W_i = np.array(weights['W_i'])
            self.b_i = np.array(weights['b_i'])
            
            self.W_c = np.array(weights['W_c'])
            self.b_c = np.array(weights['b_c'])
            self.W_o = np.array(weights['W_o'])
            self.b_o = np.array(weights['b_o'])
            
            self.W_y = np.array(weights['W_y'])
            self.b_y = np.array(weights['b_y'])
    
    def test(self, inputs: list[str], labels: list[str]):
        accuracy = 0
        probabilities = self.forward([one_hot_encode(input) for input in inputs])

        output = ''
        for i in range(len(labels)):
            prediction = idx_to_char[np.random.choice(vocab_size, p = softmax(probabilities[i].reshape(-1)))]

            output += prediction

            if prediction == labels[i]:
                accuracy += 1

        # print(f'Predictions:\nt{"".join(output)}\n')
        print(f'Accuracy: {round(accuracy * 100 / len(inputs), 2)}%')

import argparse

if __name__ == "__main__":
    hidden_size = 64
    input_size = vocab_size + hidden_size
    output_size = vocab_size

    learning_rate = 0.05
    epochs = 10
    
    lstm = LSTM(input_size, hidden_size, output_size, learning_rate)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="LSTM mode: train or test")
    parser.add_argument("--weights", type=str, help="Weights file path")

    args = parser.parse_args()
    weights_path = args.weights if args.weights else "./weights/lstm_weights.json"
    mode = args.mode if args.mode else "train"

    print(f"Running LSTM network in {mode} mode...")

    if mode == 'train':
        lstm.train(X_train, Y_train, epochs)
        lstm.save_weights(weights_path)
    elif mode == 'test':
        lstm.load_weights(weights_path)
        lstm.test(X_train, Y_train)
    elif mode == 'optimize':
        lstm.load_weights(weights_path)
        lstm.train(X_train, Y_train, epochs=10)
        lstm.save_weights(weights_path)
    else:
        print("Invalid mode. Use 'train', 'test' or 'optimize'.")