function sigmoid(z: number) {
  return 1 / (1 + Math.exp(-z));
}

// derivative of sigmoid
function sigmoid_prime(z: number) {
  return sigmoid(z) * (1 - sigmoid(z));
}

function relu(z: number) {
  return Math.max(0, z);
}

function relu_prime(z: number) {
  return z > 0 ? 1 : 0;
}

function softmax(arr) {
  const expArr = arr.map((num) => Math.exp(num));
  const sumExpArr = expArr.reduce((a, b) => a + b);
  return expArr.map((expNum) => expNum / sumExpArr);
}

function weighted_sum(inputs: number[], weights: number[]) {
  if (inputs.length !== weights.length) {
    throw new Error("inputs and weights must have the same length");
  }
  // return inputs.reduce((sum, input, index) => sum + input * weights[index], 0);
  const len = inputs.length;
  let sum = 0;
  for (let i = 0; i < len; i++) {
    sum += weights[i] * inputs[i];
  }
  return sum;
}

class Neuron {
  weights: number[];
  bias: number;
  z: number; // weighted sum + bias
  a: number; // activation, sigma(z)
  sigma = relu;
  sigma_prime = relu_prime;

  constructor(input_counts: number) {
    this.weights = Array.from({ length: input_counts }, () => Math.random());
    this.bias = Math.random();
  }

  feedforward(inputs: number[]) {
    this.z = weighted_sum(inputs, this.weights) + this.bias;
    this.a = this.sigma(this.z);
    return this.a;
  }
}

class Layer {
  neurons: Neuron[];
  A: number[];
  Z: number[];
  constructor(neuron_counts: number, input_counts: number) {
    this.neurons = Array.from({ length: neuron_counts }, () => new Neuron(input_counts));
  }

  feedforward(inputs: number[]) {
    this.A = this.neurons.map((neuron) => neuron.feedforward(inputs));
    this.Z = this.neurons.map((neuron) => neuron.z);
    return this.A;
  }
}

export class Network {
  layers: Layer[];

  constructor(layer_width: number[], input_counts: number) {
    this.layers = [];
    for (let i = 0; i < layer_width.length; i++) {
      this.layers.push(new Layer(layer_width[i], input_counts));
      input_counts = layer_width[i];
    }
  }

  predict(inputs: number[]) {
    for (let i = 0; i < this.layers.length; i++) {
      inputs = this.layers[i].feedforward(inputs);
    }
    return inputs;
  }

  backpropagate(
    inputs: number[],
    targets: number[]
  ): { nabla_b: number[][]; nabla_w: number[][][] } {
    const nabla_b = this.layers.map((layer) => layer.neurons.map(() => 0));
    const nabla_w = this.layers.map((layer) =>
      layer.neurons.map((neuron) => neuron.weights.map(() => 0))
    );
    const As = [inputs];
    const Zs = [];
    const l = this.layers.length;
    for (let i = 0; i < l; i++) {
      const layer = this.layers[i];
      layer.feedforward(inputs);
      As.push(layer.A);
      Zs.push(layer.Z);
      inputs = layer.A;
    }
    // const delta = this.layers[layer_counts - 1].neurons.map((neuron, i) => {
    //   return (neuron.a - outputs[i]) * neuron.sigma_prime(neuron.z);
    // });

    let delta = As[l].map((a, i) => {
      const y = targets[i];
      return (a - y) * sigmoid_prime(Zs[l - 1][i]);
    });
    nabla_b[l - 1] = delta;
    nabla_w[l - 1] = delta.map((d) => {
      // As[l - 1] is activation of previous layer
      return As[l - 1].map((a) => a * d);
    });
    for (let i = l - 2; i >= 0; i--) {
      const last_layer = this.layers[i + 1];
      const current_layer = this.layers[i];
      delta = current_layer.neurons.map((neuron, j) => {
        const sum = weighted_sum(
          last_layer.neurons.map((n) => n.weights[j]),
          delta
        );
        return sum * neuron.sigma_prime(neuron.z);
      });
      nabla_b[i] = delta;
      nabla_w[i] = delta.map((d) => {
        return As[i].map((a) => a * d);
      });
    }
    return { nabla_b, nabla_w };
  }

  update_mini_batch(mini_batch, eta) {
    const r = eta / mini_batch.length;
    const nabla_b = this.layers.map((layer) => layer.neurons.map(() => 0));
    const nabla_w = this.layers.map((layer) =>
      layer.neurons.map((neuron) => Array.from({ length: neuron.weights.length }, () => 0))
    );
    mini_batch.forEach((data) => {
      const { input, output } = data;
      const { nabla_b: delta_nabla_b, nabla_w: delta_nabla_w } = this.backpropagate(input, output);
      nabla_b.forEach((layer, layer_index) => {
        layer.forEach((neuron, neuron_index) => {
          nabla_b[layer_index][neuron_index] += delta_nabla_b[layer_index][neuron_index];
        });
      });
      nabla_w.forEach((layer, layer_index) => {
        layer.forEach((neuron, neuron_index) => {
          nabla_w[layer_index][neuron_index] = nabla_w[layer_index][neuron_index].map(
            (value, index) => value + delta_nabla_w[layer_index][neuron_index][index]
          );
        });
      });
    });
    this.layers.forEach((layer, layer_index) => {
      layer.neurons.forEach((neuron, neuron_index) => {
        neuron.weights = neuron.weights.map((weight, index) => {
          const w = nabla_w[layer_index][neuron_index][index];
          return weight - r * w;
        });
        const b = nabla_b[layer_index][neuron_index];
        neuron.bias -= r * b;
      });
    });
  }
}

const test_accuracy = (network, test_data) => {
  let correct = 0;
  test_data.forEach((data) => {
    const { input, output } = data;
    const prediction = network.predict(input);
    const max = Math.max(...prediction);
    const index = prediction.indexOf(max);
    if (output[index] === 1) {
      correct++;
    }
  });
  return [correct, test_data.length];
};

export function run_training(
  network: Network,
  dataset: {
    training: { input: number[]; output: number[] }[];
    test: { input: number[]; output: number[] }[];
  }
) {
  SGD(network, dataset.training, 1000, 10, 10);
  const [correct, total] = test_accuracy(network, dataset.test);
  console.log(`Test Accuracy: ${correct} / ${total} = ${((100 * correct) / total).toFixed(2)}%`);
  return network;
}

function SGD(
  network: Network,
  training_data: { input: number[]; output: number[] }[],
  epochs: number,
  mini_batch_size: number,
  eta: number
) {
  const n = training_data.length;
  for (let i = 0; i < epochs; i++) {
    shuffle(training_data);
    for (let k = 0; k < n; k += mini_batch_size) {
      const mini_batch = training_data.slice(k, k + mini_batch_size);
      network.update_mini_batch(mini_batch, eta);
    }

    console.log(`Epoch ${i}`);
  }
}

function shuffle<T extends any[]>(array: T): T {
  let counter = array.length;
  let index = 0;
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = (Math.random() * counter) | 0;
    // Decrease counter by 1
    counter--;
    // And swap the last element with it
    const temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
  return array;
}
