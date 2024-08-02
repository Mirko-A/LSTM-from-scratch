#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <fstream>
#include <nlohmann/json.hpp> // For JSON operations
#include "matrix.hpp"

using namespace std;
using json = nlohmann::json;

const string dataset_path = "datasets/shakespeare/tiny_shakespeare_small.txt";

// Utility function to read file
string read_file(const string &path)
{
    ifstream file(path);
    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    return content;
}

Matrix
init_weights(int input_size, int output_size)
{
    vector<vector<double>> weights(output_size, vector<double>(input_size));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);
    double factor = sqrt(6.0 / (input_size + output_size));

    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            weights[i][j] = dis(gen) * factor;
        }
    }

    return weights;
}

// Utility functions for activation functions
Matrix tanh(const vector<double> &x)
{
    vector<double> result(x.size());
    transform(x.begin(), x.end(), result.begin(), [](double val)
              { return std::tanh(val); });
    return result;
}

vector<double> dtanh(const vector<double> &x)
{
    vector<double> t = tanh(x);
    vector<double> result(x.size());
    transform(t.begin(), t.end(), result.begin(), [](double val)
              { return 1.0 - val * val; });
    return result;
}

vector<double> sigmoid(const vector<double> &x)
{
    vector<double> result(x.size());
    transform(x.begin(), x.end(), result.begin(), [](double val)
              { return 1.0 / (1.0 + std::exp(-val)); });
    return result;
}

vector<double> dsigmoid(const vector<double> &x)
{
    vector<double> s = sigmoid(x);
    vector<double> result(x.size());
    transform(s.begin(), s.end(), result.begin(), [](double val)
              { return val * (1.0 - val); });
    return result;
}

vector<double> softmax(const vector<double> &x)
{
    double max_val = *max_element(x.begin(), x.end());
    vector<double> exp_x(x.size());
    transform(x.begin(), x.end(), exp_x.begin(), [max_val](double val)
              { return std::exp(val - max_val); });

    double sum_exp_x = accumulate(exp_x.begin(), exp_x.end(), 0.0);
    vector<double> result(x.size());
    transform(exp_x.begin(), exp_x.end(), result.begin(), [sum_exp_x](double val)
              { return val / sum_exp_x; });

    return result;
}

// Define LSTM class
class LSTM
{
public:
    LSTM(int input_size, int hidden_size, int output_size, double learning_rate = 1e-3)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size), learning_rate(learning_rate)
    {
        W_f = init_weights(input_size, hidden_size);
        b_f = vector<vector<double>>(hidden_size, vector<double>(1, 0.0));

        W_i = init_weights(input_size, hidden_size);
        b_i = vector<vector<double>>(hidden_size, vector<double>(1, 0.0));

        W_c = init_weights(input_size, hidden_size);
        b_c = vector<vector<double>>(hidden_size, vector<double>(1, 0.0));

        W_o = init_weights(input_size, hidden_size);
        b_o = vector<vector<double>>(hidden_size, vector<double>(1, 0.0));

        W_y = init_weights(hidden_size, output_size);
        b_y = vector<vector<double>>(output_size, vector<double>(1, 0.0));

        reset_cache();
    }

    void reset_cache()
    {
        concat_inputs.clear();
        hidden_states = {-1, vector<vector<double>>(hidden_size, vector<double>(1, 0.0))};
        cell_states = {-1, vector<vector<double>>(hidden_size, vector<double>(1, 0.0))};
        forget_gates.clear();
        input_gates.clear();
        candidate_gates.clear();
        output_gates.clear();
        activation_outputs.clear();
        outputs.clear();
    }

    vector<vector<double>> forward(const vector<vector<double>> &inputs)
    {
        reset_cache();
        vector<vector<double>> output_seq;

        for (size_t t = 0; t < inputs.size(); ++t)
        {
            concat_inputs[t] = concat(hidden_states[t - 1], inputs[t]);

            forget_gates[t] = add(dot(W_f, concat_inputs[t]), b_f);
            input_gates[t] = add(dot(W_i, concat_inputs[t]), b_i);
            candidate_gates[t] = add(dot(W_c, concat_inputs[t]), b_c);
            output_gates[t] = add(dot(W_o, concat_inputs[t]), b_o);

            vector<double> fga = sigmoid(forget_gates[t]);
            vector<double> iga = sigmoid(input_gates[t]);
            vector<double> cga = tanh(candidate_gates[t]);
            vector<double> oga = sigmoid(output_gates[t]);

            cell_states[t] = add(multiply(fga, cell_states[t - 1]), multiply(iga, cga));
            hidden_states[t] = multiply(oga, tanh(cell_states[t]));

            vector<double> output = softmax(add(dot(W_y, hidden_states[t]), b_y));
            output_seq.push_back(output);
            outputs[t] = output;
        }

        return output_seq;
    }

    void backward(const vector<vector<double>> &labels)
    {
        // Implementation of backward pass
    }

    vector<double> cross_entropy_loss(const vector<double> &y, const vector<double> &y_ref)
    {
        vector<double> loss(y.size());
        transform(y.begin(), y.end(), y_ref.begin(), loss.begin(), [](double yi, double y_refi)
                  { return -y_refi * std::log(yi); });
        return loss;
    }

    void train(const string &inputs, const string &labels, int epochs = 1)
    {
        vector<vector<double>> one_hot_inputs(inputs.size(), vector<double>(vocab_size, 0.0));
        vector<vector<double>> one_hot_labels(labels.size(), vector<double>(vocab_size, 0.0));

        // Convert inputs and labels to one-hot encoding here
        // ...

        vector<double> losses;
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            vector<vector<double>> predictions = forward(one_hot_inputs);
            double loss = 0;
            for (size_t i = 0; i < predictions.size(); ++i)
            {
                loss += accumulate(cross_entropy_loss(predictions[i], one_hot_labels[i]).begin(), cross_entropy_loss(predictions[i], one_hot_labels[i]).end(), 0.0);
            }
            losses.push_back(loss / predictions.size());
            backward(one_hot_labels);
        }

        // Save weights
        save_weights("weights/lstm_weights.json");
    }

    void test(const string &inputs, const string &labels)
    {
        vector<vector<double>> one_hot_inputs(inputs.size(), vector<double>(vocab_size, 0.0));
        vector<vector<double>> one_hot_labels(labels.size(), vector<double>(vocab_size, 0.0));

        // Convert inputs and labels to one-hot encoding here
        // ...

        vector<vector<double>> probabilities = forward(one_hot_inputs);
        string output;
        double accuracy = 0;

        for (size_t i = 0; i < labels.size(); ++i)
        {
            int prediction_idx = random_choice(probabilities[i]);
            output += idx_to_char[prediction_idx];

            if (output.back() == labels[i])
            {
                accuracy += 1;
            }
        }

        accuracy = (accuracy / inputs.size()) * 100;
        cout << "Predictions:\n"
             << output << endl;
        cout << "Accuracy: " << accuracy << "%" << endl;
    }

    void generate(const string &prompt, int length = 128)
    {
        vector<vector<double>> one_hot_prompt(prompt.size(), vector<double>(vocab_size, 0.0));

        // Convert prompt to one-hot encoding here
        // ...

        string output = prompt;
        for (int i = 0; i < length; ++i)
        {
            vector<vector<double>> probabilities = forward(one_hot_prompt);
            int prediction_idx = random_choice(probabilities.back());
            output += idx_to_char[prediction_idx];
            one_hot_prompt.push_back(one_hot_encode(output.back()));
        }

        cout << output << endl;
    }

    void save_weights(const string &weights_path)
    {
        json weights = {
            {"W_f", W_f},
            {"b_f", b_f},
            {"W_i", W_i},
            {"b_i", b_i},
            {"W_c", W_c},
            {"b_c", b_c},
            {"W_o", W_o},
            {"b_o", b_o},
            {"W_y", W_y},
            {"b_y", b_y}};

        ofstream file(weights_path);
        file << weights.dump(4);
    }

    void load_weights(const string &weights_path)
    {
        ifstream file(weights_path);
        json weights;
        file >> weights;

        W_f = weights["W_f"].get<vector<vector<double>>>();
        b_f = weights["b_f"].get<vector<vector<double>>>();
        W_i = weights["W_i"].get<vector<vector<double>>>();
        b_i = weights["b_i"].get<vector<vector<double>>>();
        W_c = weights["W_c"].get<vector<vector<double>>>();
        b_c = weights["b_c"].get<vector<vector<double>>>();
        W_o = weights["W_o"].get<vector<vector<double>>>();
        b_o = weights["b_o"].get<vector<vector<double>>>();
        W_y = weights["W_y"].get<vector<vector<double>>>();
        b_y = weights["b_y"].get<vector<vector<double>>>();
    }

private:
    int input_size;
    int hidden_size;
    int output_size;
    double learning_rate;

    vector<vector<double>> W_f, W_i, W_c, W_o, W_y;
    vector<vector<double>> b_f, b_i, b_c, b_o, b_y;

    unordered_map<int, vector<vector<double>>> concat_inputs;
    unordered_map<int, vector<vector<double>>> hidden_states;
    unordered_map<int, vector<vector<double>>> cell_states;
    unordered_map<int, vector<vector<double>>> forget_gates;
    unordered_map<int, vector<vector<double>>> input_gates;
    unordered_map<int, vector<vector<double>>> candidate_gates;
    unordered_map<int, vector<vector<double>>> output_gates;
    unordered_map<int, vector<vector<double>>> activation_outputs;
    unordered_map<int, vector<vector<double>>> outputs;

    vector<double> one_hot_encode(char c)
    {
        vector<double> one_hot(vocab_size, 0.0);
        one_hot[char_to_idx[c]] = 1.0;
        return one_hot;
    }

    int random_choice(const vector<double> &probs)
    {
        random_device rd;
        mt19937 gen(rd());
        discrete_distribution<> d(probs.begin(), probs.end());
        return d(gen);
    }

    vector<double> add(const vector<double> &a, const vector<double> &b)
    {
        vector<double> result(a.size());
        transform(a.begin(), a.end(), b.begin(), result.begin(), [](double ai, double bi)
                  { return ai + bi; });
        return result;
    }

    vector<double> multiply(const vector<double> &a, const vector<double> &b)
    {
        vector<double> result(a.size());
        transform(a.begin(), a.end(), b.begin(), result.begin(), [](double ai, double bi)
                  { return ai * bi; });
        return result;
    }

    vector<double> dot(const vector<vector<double>> &mat, const vector<double> &vec)
    {
        vector<double> result(mat.size());
        for (size_t i = 0; i < mat.size(); ++i)
        {
            result[i] = inner_product(mat[i].begin(), mat[i].end(), vec.begin(), 0.0);
        }
        return result;
    }

    vector<double> concat(const vector<double> &a, const vector<double> &b)
    {
        vector<double> result(a.size() + b.size());
        copy(a.begin(), a.end(), result.begin());
        copy(b.begin(), b.end(), result.begin() + a.size());
        return result;
    }
};

int main(int argc, char **argv)
{
    int hidden_size = 64;
    int vocab_size = 100; // This should be set to the actual vocabulary size
    int input_size = vocab_size + hidden_size;
    int output_size = vocab_size;
    double learning_rate = 0.06;
    int epochs = 1;

    LSTM lstm(input_size, hidden_size, output_size, learning_rate);

    string mode = "train";
    string weights_path = "./weights/lstm_weights.json";

    // Parse command line arguments here (using libraries like `getopt` or `boost::program_options`)
    // ...

    string data = read_file(dataset_path);
    vector<string> X_train = {data.substr(0, data.size() - 1)};
    vector<string> Y_train = {data.substr(1)};

    cout << "Running LSTM network in '" << mode << "' mode..." << endl;

    if (mode == "train")
    {
        lstm.train(X_train[0], Y_train[0], epochs);
        lstm.save_weights(weights_path);
    }
    else if (mode == "test")
    {
        lstm.load_weights(weights_path);
        lstm.test(X_train[0], Y_train[0]);
    }
    else if (mode == "optimize")
    {
        lstm.load_weights(weights_path);
        lstm.train(X_train[0], Y_train[0], epochs);
        lstm.save_weights(weights_path);
    }
    else if (mode == "generate")
    {
        lstm.load_weights(weights_path);
        cout << lstm.generate("The ", 512) << endl;
    }
    else
    {
        cout << "Invalid mode. Use 'train', 'test', 'optimize' or 'generate'." << endl;
    }

    return 0;
}
