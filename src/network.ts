function matrix(n: number, m: number, init: (n: number, m: number) => number) {
  return Array.from({ length: n }, (_, n) => Array.from({ length: m }, (_, m) => init(n, m)));
}

export class Network {
  num_layers: number;
  sizes: number[];
  biases: number[][];
  weights: number[][][];

  constructor(sizes: number[]) {
    this.num_layers = sizes.length;
    // sizes[0] is the number of raw inputs
    this.sizes = sizes;
    // both biases and weights' length is 1 shorter than sizes
    // because the first input layer has no biases or weights
    // only hidden layers and output layer have biases and weights
    // each bias vector is a j*1 matrix, when j is the number of neurons in the current layer
    this.biases = sizes.slice(1).map((n) => Array.from({ length: n }, () => Math.random()));
    // each weight matrix is a n*m matrix, when n is the number of neurons in the current layer
    // and m is the number of neurons in the previous layer
    this.weights = sizes.slice(0, -1).map((m, l) => {
      return matrix(sizes[l + 1], m, () => Math.random());
    });
  }

  feedforward(a) {
    this.biases.forEach((b, i) => {
      let z = np.dot(this.weights[i], a);
      a = sigmoid([...z].map((num, j) => num + b[j]));
    });
    return a;
  }

  SGD(training_data, epochs, mini_batch_size, eta, test_data = null) {
    if (test_data) {
      var n_test = test_data.length;
    }
    var n = training_data.length;
    for (var j = 0; j < epochs; j++) {
      shuffleArrayInPlace(training_data);
      for (var k = 0; k < n; k += mini_batch_size) {
        let mini_batch = training_data.slice(k, k + mini_batch_size);
        this.update_mini_batch(mini_batch, eta);
      }
      if (test_data) {
        console.log("Epoch " + j + ": " + this.evaluate(test_data) + " / " + n_test);
      } else {
        console.log("Epoch " + j + " complete");
      }
    }
  }

  update_mini_batch(mini_batch, eta) {
    let nabla_b = this.biases.map((b) => new Array(b.length).fill(0));
    let nabla_w = this.weights.map((layer) =>
      Array.from(new Array(layer.length), () => new Array(layer[0].length).fill(0))
    );
    mini_batch.forEach(([x, y]) => {
      let { delta_nabla_b, delta_nabla_w } = this.backprop(x, y);
      nabla_b = nabla_b.map((nb, i) => nb.zip(delta_nabla_b[i], (a, b) => a + b));
      nabla_w = nabla_w.map((nw, i) => nw.zip(delta_nabla_w[i], (a, b) => a + b));
    });
    this.weights = this.weights.zip(nabla_w, (w, nw) =>
      w.zip(nw, (wi, nwi) => wi.zip(nwi, (a, b) => a - (eta / mini_batch.length) * b))
    );
    this.biases = this.biases.zip(nabla_b, (b, nb) =>
      b.zip(nb, (bi, nbi) => bi + (eta / mini_batch.length) * nbi)
    );
  }

  backprop(x, y) {
    var nabla_b = this.biases.map((b) => new Array(b.length).fill(0));
    var nabla_w = this.weights.map((layer) =>
      Array.from(new Array(layer.length), () => new Array(layer[0].length).fill(0))
    );
    var activation = x;
    var activations = [x];
    var zs = [];
    this.biases.forEach((b, i) => {
      let z = np.dot(this.weights[i], activation);
      zs.push(z);
      activation = sigmoid([...z].map((num, j) => num + b[j]));
      activations.push(activation);
    });
    let delta = cost_derivative(activations[activations.length - 1], y).map(
      (d, j) => d * sigmoid_prime(zs[zs.length - 1][j])
    );
    nabla_b[nabla_b.length - 1] = delta;
    nabla_w[nabla_w.length - 1] = np.dot(delta, activations[activations.length - 2].transpose());

    for (var l = 2; l < this.num_layers; l++) {
      let z = zs[zs.length - l];
      let sp = sigmoid_prime([...z]);
      delta = np.dot(this.weights[-l + 1].transpose(), delta).map((d, j) => d * sp[j]);
      nabla_b[-l] = delta;
      nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose());
    }

    return { nabla_b, nabla_w };
  }

  evaluate(test_data) {
    let test_results = test_data.map(([x, y]) => [np.argmax(this.feedforward(x)), y]);
    let num_correct = sum(test_results.map(([x, y]) => Number(x === y)));
    return num_correct;
  }

  cost_derivative(output_activations, y) {}
}
