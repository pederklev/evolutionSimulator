import { NeuralNetwork } from "../NeuroEvolution/NeuralNetwork";
import * as MatterJS from "matter-js";
import * as tfjs from "@tensorflow/tfjs";

export interface RunnerParams {
  id?: number;
  leftLegLength?: number;
  leftLegWidth?: number;
  rightLegLength?: number;
  rightLegWidth?: number;
  bodyLength?: number;
  bodyWidth?: number;
  posX?: number;
  posY?: number;
  canvasWidth: number;
  canvasHeight: number;
}

export class BipedalRunner {
  constructor(params: RunnerParams) {
    this.id = params.id || 0;
    this.leftLegLength = params.leftLegLength || 60;
    this.leftLegWidth = params.leftLegWidth || 10;
    this.rightLegLength = params.rightLegLength || 60;
    this.rightLegWidth = params.rightLegWidth || 10;
    this.bodyLength = params.bodyLength || 100;
    this.bodyWidth = params.bodyWidth || 20;
    this.posX = params.posX || params.canvasWidth * 0.1;
    this.posY = params.posY || params.canvasHeight * 0.8;
    this.score = 0;
    this.fitness = 0;
    this.brain = new NeuralNetwork(10, 25, 2);

    this.canvasWidth = params.canvasWidth || 800;
    this.canvasHeight = params.canvasHeight || 600;

    ////////////////
    // Body parts //
    ////////////////
    this.leftLeg = MatterJS.Bodies.rectangle(
      this.posX - 75,
      this.posY + 40,
      this.leftLegLength,
      this.leftLegWidth,
      {
        friction: 1,
        restitution: 0.1,
        density: 0.05,
        collisionFilter: {
          category: 0x0006,
          mask: 0x0001,
          group: -1
        }
        // isStatic: true,
      }
    );

    this.rightLeg = MatterJS.Bodies.rectangle(
      this.posX + 80,
      this.posY + 40,
      this.rightLegLength,
      this.rightLegWidth,
      {
        friction: 1,
        restitution: 0.1,
        density: 0.05,
        collisionFilter: {
          category: 0x0004,
          mask: 0x0001,
          group: -1
        }
        // isStatic: true,
      }
    );

    this.body = MatterJS.Bodies.rectangle(
      this.posX,
      this.posY,
      this.bodyLength,
      this.bodyWidth,
      {
        friction: 1,
        restitution: 0.1,
        density: 0.05,
        collisionFilter: {
          category: 0x0002,
          mask: 0x0001,
          group: -1
        }
        // isStatic: true,
      }
    );

    this.leftJoint = MatterJS.Constraint.create({
      bodyA: this.leftLeg,
      bodyB: this.body,
      pointA: { x: (this.leftLegLength / 2) * 0.8, y: 0 },
      pointB: { x: (-this.bodyLength / 2) * 0.8, y: 0 },
      length: 0,
      stiffness: 1
    });

    this.rightJoint = MatterJS.Constraint.create({
      bodyA: this.rightLeg,
      bodyB: this.body,
      pointA: { x: (-this.rightLegLength / 2) * 0.8, y: 0 },
      pointB: { x: (this.bodyLength / 2) * 0.8, y: 0 },
      length: 0,
      stiffness: 1
    });

    ///////////////////
    // Muscle Joints //
    ///////////////////
    this.leftMuscle = MatterJS.Constraint.create({
      bodyA: this.leftLeg,
      bodyB: this.body,
      length: 0.8 * (this.rightLegLength / 2 + this.bodyLength / 2),
      pointA: { x: (-this.rightLegLength / 2) * 0.8, y: 0 },
      stiffness: 1
    });

    this.rightMuscle = MatterJS.Constraint.create({
      bodyA: this.rightLeg,
      bodyB: this.body,
      length: 0.8 * (this.rightLegLength / 2 + this.bodyLength / 2),
      pointA: { x: (this.rightLegLength / 2) * 0.8, y: 0 },
      stiffness: 1
    });
  }

  id: number;
  leftLegLength: number;
  leftLegWidth: number;
  rightLegLength: number;
  rightLegWidth: number;
  bodyLength: number;
  bodyWidth: number;
  posX: number;
  posY: number;
  score: number;
  fitness: number;
  brain: any;

  leftLeg: MatterJS.Body;
  leftJoint: MatterJS.Constraint;
  leftMuscle: MatterJS.Constraint;

  rightLeg: MatterJS.Body;
  rightJoint: MatterJS.Constraint;
  rightMuscle: MatterJS.Constraint;

  body: MatterJS.Body;

  canvasWidth: number;
  canvasHeight: number;

  makeJoints() {
    ////////////
    // Joints //
    ////////////
    this.leftJoint = MatterJS.Constraint.create({
      bodyA: this.leftLeg,
      bodyB: this.body,
      pointA: { x: (this.leftLegLength / 2) * 0.8, y: 0 },
      pointB: { x: (-this.bodyLength / 2) * 0.8, y: 0 },
      length: 0,
      stiffness: 1
    });

    this.rightJoint = MatterJS.Constraint.create({
      bodyA: this.rightLeg,
      bodyB: this.body,
      pointA: { x: (-this.rightLegLength / 2) * 0.8, y: 0 },
      pointB: { x: (this.bodyLength / 2) * 0.8, y: 0 },
      length: 0,
      stiffness: 1
    });

    ///////////////////
    // Muscle Joints //
    ///////////////////
    this.leftMuscle = MatterJS.Constraint.create({
      bodyA: this.leftLeg,
      bodyB: this.body,
      length: 0.8 * (this.rightLegLength / 2 + this.bodyLength / 2),
      pointA: { x: (-this.rightLegLength / 2) * 0.8, y: 0 },
      stiffness: 1
    });

    this.rightMuscle = MatterJS.Constraint.create({
      bodyA: this.rightLeg,
      bodyB: this.body,
      length: 0.8 * (this.rightLegLength / 2 + this.bodyLength / 2),
      pointA: { x: (this.rightLegLength / 2) * 0.8, y: 0 },
      stiffness: 1
    });
  }

  /**
   * Adds all parts of the person in MatterJS world
   * @param {Matter.World} world
   */
  addToWorld(world: MatterJS.World) {
    MatterJS.World.add(world, [this.leftLeg, this.rightLeg, this.body]);
    MatterJS.World.add(world, [this.leftJoint]);
    MatterJS.World.add(world, [this.leftMuscle]);
    MatterJS.World.add(world, [this.rightJoint]);
    MatterJS.World.add(world, [this.rightMuscle]);
  }

  /**
   * Removes all parts of the person from MatterJS world
   * @param {Matter.World} world
   */
  removeFromWorld(world: MatterJS.World) {
    MatterJS.World.remove(world, this.leftLeg);
    MatterJS.World.remove(world, this.rightLeg);
    MatterJS.World.remove(world, this.body);
    MatterJS.World.remove(world, this.leftJoint);
    MatterJS.World.remove(world, this.leftMuscle);
    MatterJS.World.remove(world, this.rightJoint);
    MatterJS.World.remove(world, this.rightMuscle);

    // Dispose its brain
    this.brain.dispose();
  }

  /**
   * Returns an object of with all the parameters required to create a new Bipedal
   * @returns {Object}
   */
  getParams() {
    return {
      id: this.id,
      leftLegLength: this.leftLegLength,
      leftLegWidth: this.leftLegWidth,
      rightLegLength: this.rightLegLength,
      rightLegWidth: this.rightLegWidth,
      bodyLength: this.bodyLength,
      bodyWidth: this.bodyWidth,
      posX: this.posX,
      posY: this.posY,
      canvasWidth: this.canvasWidth,
      canvasHeight: this.canvasHeight
    };
  }

  clone() {
    const params = this.getParams();

    let bipedal = new BipedalRunner(params);
    bipedal.brain.dispose();
    bipedal.brain = this.brain.clone();
    return bipedal;
  }

  adjustScore() {
    // Balancing score (Head should be level)
    const isHeadBalanced = Math.abs(this.body.angle) < 0.2;
    const walkingScore = this.body.position.x - this.posX;
    const velocity = this.body.velocity.x;
    this.score += walkingScore * (isHeadBalanced ? 2 : 0.5) * velocity;
  }

  think(boundary: any) {
    // Prepare inputs
    const ground = boundary.ground;
    const bodyHeightAboveGround =
      (ground.position.y - this.body.position.y) / this.canvasWidth;
    const leftLegHeightAboveGround =
      (ground.position.y - this.leftLeg.position.y) / this.canvasWidth;
    const rightLegHeightAboveGround =
      (ground.position.y - this.rightLeg.position.y) / this.canvasWidth;
    const vx = this.body.velocity.x;
    const vy = this.body.velocity.y;
    const leftMuscleLength = this.leftMuscle.length / 70;
    const rightMuscleLength = this.rightMuscle.length / 70;
    const bodyAngle = this.body.angle;
    const leftLegAngle = this.leftLeg.angle;
    const rightLegAngle = this.rightLeg.angle;

    const input = [
      bodyHeightAboveGround,
      leftLegHeightAboveGround,
      rightLegHeightAboveGround,
      vx,
      vy,
      leftMuscleLength,
      rightMuscleLength,
      bodyAngle,
      leftLegAngle,
      rightLegAngle
    ];

    // Predict
    const result = this.brain.predict(input);

    // Move Muscles
    let leftMuscleShift = result[0] > 0.5 ? 2 : -2;
    let rightMuscleShift = result[1] > 0.5 ? 2 : -2;

    if (leftMuscleShift < 0 && this.leftMuscle.length + leftMuscleShift > 25)
      this.leftMuscle.length += leftMuscleShift;
    if (leftMuscleShift > 0 && this.leftMuscle.length + leftMuscleShift < 70)
      this.leftMuscle.length += leftMuscleShift;

    if (rightMuscleShift < 0 && this.rightMuscle.length + rightMuscleShift > 25)
      this.rightMuscle.length += rightMuscleShift;
    if (rightMuscleShift > 0 && this.rightMuscle.length + rightMuscleShift < 70)
      this.rightMuscle.length += rightMuscleShift;

    // Adjust Score
    this.adjustScore();
  }

  crossover(partner: BipedalRunner) {
    let parentA_in_dna = this.brain.input_weights.dataSync();
    let parentA_out_dna = this.brain.output_weights.dataSync();
    let parentB_in_dna = partner.brain.input_weights.dataSync();
    let parentB_out_dna = partner.brain.output_weights.dataSync();

    let mid = Math.floor(Math.random() * parentA_in_dna.length);
    let child_in_dna = [
      ...parentA_in_dna.slice(0, mid),
      ...parentB_in_dna.slice(mid, parentB_in_dna.length)
    ];
    let child_out_dna = [
      ...parentA_out_dna.slice(0, mid),
      ...parentB_out_dna.slice(mid, parentB_out_dna.length)
    ];

    let child = this.clone();
    let input_shape = this.brain.input_weights.shape;
    let output_shape = this.brain.output_weights.shape;

    child.brain.dispose();

    child.brain.input_weights = tfjs.tensor(child_in_dna, input_shape);
    child.brain.output_weights = tfjs.tensor(child_out_dna, output_shape);

    return child;
  }

  mutate() {
    function fn(x: number) {
      // if (random(1) < 0.1) {
      if (Math.random() < 0.1) {
        // let offset = randomGaussian() * 0.5;
        let offset = Math.random() * 0.5;
        let newx = x + offset;
        return newx;
      }
      return x;
    }

    let ih = this.brain.input_weights.dataSync().map(fn);
    let ih_shape = this.brain.input_weights.shape;
    this.brain.input_weights.dispose();
    this.brain.input_weights = tfjs.tensor(ih, ih_shape);

    let ho = this.brain.output_weights.dataSync().map(fn);
    let ho_shape = this.brain.output_weights.shape;
    this.brain.output_weights.dispose();
    this.brain.output_weights = tfjs.tensor(ho, ho_shape);
  }
}
