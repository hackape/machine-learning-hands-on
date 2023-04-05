import "./app.css";
import App from "./App.svelte";
// import { train, predict } from "./train";
import mnist from "mnist";
import { Network, run_training } from "./tune";

const app = new App({
  target: document.getElementById("app"),
});

export default app;

// const model = await train();
// window["model"] = model;
// window["predict"] = predict;
// console.log("predict", predict(mnist[9].get()));

window["mnist"] = mnist;
window["network"] = window["model"] = new Network([2, 1], 2);
window["run_training"] = run_training;

function main() {
  run_training(window["network"], {
    training: [...gen_data_OR(5000)],
    test: [...gen_data_OR(1000)],
  });
}

window["main"] = main;
main();

// generate boolean logic data

function gen_data_AND(size: number) {
  return Array.from({ length: size }, () => {
    const a = Math.random() > 0.5 ? 1 : 0;
    const b = Math.random() > 0.5 ? 1 : 0;
    const c = a && b;
    return { input: [a, b], output: [c] };
  });
}

function gen_data_OR(size: number) {
  return Array.from({ length: size }, () => {
    const a = Math.random() > 0.5 ? 1 : 0;
    const b = Math.random() > 0.5 ? 1 : 0;
    const c = a || b;
    return { input: [a, b], output: [c] };
  });
}

function gen_data_XOR(size: number) {
  return Array.from({ length: size }, () => {
    const a = Math.random() > 0.5 ? 1 : 0;
    const b = Math.random() > 0.5 ? 1 : 0;
    const c = a ^ b;
    return { input: [a, b], output: [c] };
  });
}
