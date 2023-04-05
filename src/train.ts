import * as tf from "@tensorflow/tfjs";
import mnist from "mnist";

window.tf = tf;

const dataset = mnist.set(1000, 100);
const xTrain = tf.tensor(dataset.training.map((data) => data.input));
const yTrain = tf.tensor(dataset.training.map((data) => data.output));
const xTest = tf.tensor(dataset.test.map((data) => data.input));
const yTest = tf.tensor(dataset.test.map((data) => data.output));

const model = tf.sequential();

model.add(
  tf.layers.dense({
    units: 32,
    activation: "relu",
    inputShape: [784],
  })
);

model.add(
  tf.layers.dense({
    units: 10,
    activation: "softmax",
  })
);
model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

export async function train() {
  console.log("start training...");
  const history = await model.fit(xTrain, yTrain, {
    epochs: 10,
    validationData: [xTest, yTest],
  });
  const result = model.evaluate(xTest, yTest);
  console.log(`Test accuracy: ${result[1]}`);
  console.log("training history:", history);
  return model;
}

export function predict(input: number[]) {
  const output = model.predict(tf.tensor(input).reshape([1, 784])) as tf.Tensor;
  console.log("prediction:", output);
  return mnist.toNumber(output.arraySync()[0]);
}
