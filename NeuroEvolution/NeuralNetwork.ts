import * as tfjs from "@tensorflow/tfjs";
import { Rank, Tensor } from "@tensorflow/tfjs";
import { TensorLike } from "@tensorflow/tfjs-core/dist/types";

interface JSONData {
  input_nodes: number;
  input_weights: TensorLike;

  hidden_nodes: number;

  output_nodes: number;
  output_weights: TensorLike;
}

/** Simple Neural Network library that can only create neural networks of exactly 3 layers */
export class NeuralNetwork {
  constructor(input_nodes: number, hidden_nodes: number, output_nodes: number) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    // Initialize random weights
    this.input_weights = tfjs.randomNormal([
      this.input_nodes,
      this.hidden_nodes
    ]);
    this.output_weights = tfjs.randomNormal([
      this.hidden_nodes,
      this.output_nodes
    ]);
  }

  input_nodes: number;
  input_weights: Tensor<Rank>;

  hidden_nodes: number;

  output_nodes: number;
  output_weights: Tensor<Rank>;

  /**
   * Takes in a 1D array and feed forwards through the network
   * @param {array} - Array of inputs
   */
  predict(user_input: number[]) {
    let output;
    tfjs.tidy(() => {
      /* Takes a 1D array */
      let input_layer = tfjs.tensor(user_input, [1, this.input_nodes]);
      let hidden_layer = input_layer.matMul(this.input_weights).sigmoid();
      let output_layer = hidden_layer.matMul(this.output_weights).sigmoid();
      output = output_layer.dataSync();
    });
    return output;
  }

  /**
   * Returns a new network with the same weights as this Neural Network
   * @returns {NeuralNetwork}
   */
  clone() {
    let clonie = new NeuralNetwork(
      this.input_nodes,
      this.hidden_nodes,
      this.output_nodes
    );
    clonie.dispose();
    clonie.input_weights = tfjs.clone(this.input_weights);
    clonie.output_weights = tfjs.clone(this.output_weights);
    return clonie;
  }

  /**
   * Dispose the input and output weights from the memory
   */
  dispose() {
    this.input_weights.dispose();
    this.output_weights.dispose();
  }

  /**
   * Save as JSON
   */
  toJson() {
    let json_data = {
      input_nodes: this.input_nodes,
      hidden_nodes: this.hidden_nodes,
      output_nodes: this.output_nodes,
      input_weights: this.input_weights.dataSync(),
      output_weights: this.output_weights.dataSync()
    };
    return JSON.stringify(json_data);
  }

  /**
   * Load model from JSON
   * @param {json} json_data
   */
  loadFromJson(json_data: JSONData) {
    this.input_nodes = json_data.input_nodes;
    this.hidden_nodes = json_data.hidden_nodes;
    this.output_nodes = json_data.output_nodes;
    let input_weights = json_data.input_weights;
    let output_weights = json_data.output_weights;
    this.input_weights = tfjs.tensor(input_weights, [
      json_data.input_nodes,
      json_data.hidden_nodes
    ]);
    this.output_weights = tfjs.tensor(output_weights, [
      json_data.hidden_nodes,
      json_data.output_nodes
    ]);
  }
}
