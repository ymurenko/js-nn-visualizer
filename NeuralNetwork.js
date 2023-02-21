const elS = document.querySelector(".network");
const avgErrStat = document.querySelector(".avg-err");
const numIterStat = document.querySelector(".num-iter");
const l0 = document.querySelector(".l0").children;
const l1 = document.querySelector(".l1").children;
const l2 = document.querySelector(".l2").children;
const l3 = document.querySelector(".l3").children;
const l4 = document.querySelector(".l4").children;
const UPDATE_INTERVAL = 50;
const TRAINING_ITERATIONS = 500;

function calcDelta(error, layer) {
  return math.dotMultiply(
    error,
    math.map(layer, (a) => sigmoid_derivative(a))
  );
}

function calcError(layerDelta, weights) {
  return math.multiply(layerDelta, math.transpose(weights));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-1 * x));
}

function sigmoid_derivative(x) {
  return x * (1 - x);
}

function getSvgLineLength(x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  return Math.sqrt(dx * dx + dy * dy);
}

function drawSvgTriangles(svgElement, startId, endId, modifier, weight) {
  let startDiv;
  let endDiv;
  if (weight > 0) {
    startDiv = document.querySelector(startId);
    endDiv = document.querySelector(endId);
  }
  if (weight < 0) {
    startDiv = document.querySelector(endId);
    endDiv = document.querySelector(startId);
  }
  if (weight !== 0) {
    const startX = startDiv.offsetLeft + startDiv.offsetWidth / 2;
    const startY = startDiv.offsetTop + startDiv.offsetHeight / 2;
    const endX = endDiv.offsetLeft + endDiv.offsetWidth / 2;
    const endY = endDiv.offsetTop + endDiv.offsetHeight / 2;

    const distanceModifier = getSvgLineLength(startX, startY, endX, endY) / 450;
    const width =  (55 * distanceModifier) / (modifier * Math.abs(weight));
    const length = Math.sqrt((endX - startX) ** 2 + (endY - startY) ** 2);
    
    const angle = Math.atan2(endY - startY, endX - startX);
    const angleOffset = Math.PI / 6; // adjust this to change the angle of the triangle

    const x1 = startX;
    const y1 = startY;
    const x2 = endX - (length / width) * Math.cos(angle + angleOffset);
    const y2 = endY - (length / width) * Math.sin(angle + angleOffset);
    const x3 = endX - (length / width) * Math.cos(angle - angleOffset);
    const y3 = endY - (length / width) * Math.sin(angle - angleOffset);

    const opacity = Math.max(Math.abs(modifier * weight), 0.15);

    const polygon = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "polygon"
    );
    polygon.setAttribute("points", `${x1},${y1} ${x2},${y2} ${x3},${y3}`);
    polygon.setAttribute(
      "style",
      `fill: black; 
       stroke: black;
       stroke-width: 3px;
       opacity: ${opacity};`
    );

    svgElement.appendChild(polygon);
  }
}

function drawSvgLines(svgElement, startId, endId, modifier, weight) {
  const startDiv = document.querySelector(startId);
  const endDiv = document.querySelector(endId);

  const startX = startDiv.offsetLeft + startDiv.offsetWidth / 2;
  const startY = startDiv.offsetTop + startDiv.offsetHeight / 2;
  const endX = endDiv.offsetLeft + endDiv.offsetWidth / 2;
  const endY = endDiv.offsetTop + endDiv.offsetHeight / 2;

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", startX);
  line.setAttribute("y1", startY);
  line.setAttribute("x2", endX);
  line.setAttribute("y2", endY);
  line.setAttribute(
    "style",
    `stroke: black;stroke-width: ${Math.max(
      Math.abs(weight) * 10,
      3
    )}px; opacity: ${Math.max(Math.abs(modifier * weight), 0.15)};`
  );
  svgElement.appendChild(line);
}

function getOutputColor(value) {
  const color1 = [0, 255, 0];
  const color2 = [255, 0, 0];
  const blend = (value - 1) / 9;
  const r = Math.round(color1[0] * (1 - blend) + color2[0] * blend);
  const g = Math.round(color1[1] * (1 - blend) + color2[1] * blend);
  const b = Math.round(color1[2] * (1 - blend) + color2[2] * blend);
  return `rgb(${r},${g},${b})`;
}

function getStrokeColor(value) {
  const color1 = [3, 136, 252];
  const color2 = [252, 61, 3];
  const blend = (value + 1) / 3;
  const r = Math.round(color1[0] * (1 - blend) + color2[0] * blend);
  const g = Math.round(color1[1] * (1 - blend) + color2[1] * blend);
  const b = Math.round(color1[2] * (1 - blend) + color2[2] * blend);
  return `rgb(${r},${g},${b})`;
}

function getPerceptronColor(value) {
  const color1 = [3, 136, 252];
  const color2 = [252, 61, 3];
  const blend = value;
  const r = Math.round(color1[0] * (1 - blend) + color2[0] * blend);
  const g = Math.round(color1[1] * (1 - blend) + color2[1] * blend);
  const b = Math.round(color1[2] * (1 - blend) + color2[2] * blend);
  return `rgb(${r},${g},${b})`;
}

function setLabel(parent, classname, text) {
  const oldLabel = document.querySelector(`.${classname}`);
  if (oldLabel) {
    oldLabel.remove();
  }
  const label = document.createElement("p");
  label.setAttribute("class", `label ${classname}`);
  label.textContent = text;
  parent.appendChild(label);
}

function refreshElement(el) {
  el.style.display = "none";
  el.style.display = "block";
}

function setAvgErr(value) {
  avgErrStat.textContent = value;
}

function setNumIter(value) {
  numIterStat.textContent = value;
}

const input_data = [
  [1, 0, 0, 0],
  [1, 1, 0, 0],
  [1, 1, 1, 0],
  [1, 1, 1, 1],
  [0, 1, 1, 1],
  [0, 0, 1, 1],
  [0, 0, 0, 1],
  [0, 0, 0, 0],
  [0, 1, 1, 0],
  [1, 0, 0, 1],
  [0, 1, 0, 1],
  [1, 0, 1, 0],
  [0, 0, 1, 0],
  [0, 1, 0, 0],
];

const expected_output = [
  [0.1],
  [0.2],
  [0.3],
  [0.4],
  [0.3],
  [0.2],
  [0.1],
  [0.0],
  [0.2],
  [0.2],
  [0.2],
  [0.2],
  [0.1],
  [0.1],
];

const input_data2 = [
  [[1, 0, 0, 0]],
  [[1, 1, 0, 0]],
  [[1, 1, 1, 0]],
  [[1, 1, 1, 1]],
  [[0, 1, 1, 1]],
  [[0, 0, 1, 1]],
  [[0, 0, 0, 1]],
  [[0, 0, 0, 0]],
  [[0, 1, 1, 0]],
  [[1, 0, 0, 1]],
  [[0, 1, 0, 1]],
  [[1, 0, 1, 0]],
  [[0, 0, 1, 0]],
  [[0, 1, 0, 0]],
];

const expected_output2 = [
  [[0.1]],
  [[0.2]],
  [[0.3]],
  [[0.4]],
  [[0.3]],
  [[0.2]],
  [[0.1]],
  [[0.0]],
  [[0.2]],
  [[0.2]],
  [[0.2]],
  [[0.2]],
  [[0.1]],
  [[0.1]],
];

class NeuralNetwork {
  constructor(num_inputs, num_outputs, num_hidden_nodes, dataset_length) {
    this.num_iter = 1;
    this.sum_error = 1;
    this.num_inputs = num_inputs;
    this.num_outputs = num_outputs;
    this.num_hidden_nodes = num_hidden_nodes;
    this.dataset_length = dataset_length;
    this.weights1 = Array(num_inputs)
      .fill(null)
      .map(() =>
        Array(num_hidden_nodes)
          .fill(null)
          .map(() => Math.random() * 2 - 1)
      );
    this.weights2 = Array(num_hidden_nodes)
      .fill(null)
      .map(() =>
        Array(num_hidden_nodes)
          .fill(null)
          .map(() => Math.random() * 2 - 1)
      );
    this.weights3 = Array(num_hidden_nodes)
      .fill(null)
      .map(() =>
        Array(num_outputs)
          .fill(null)
          .map(() => Math.random() * 2 - 1)
      );
  }

  feedforward(input_data, print_output) {
    this.layer0 = input_data;
    this.layer1 = math.map(math.multiply(this.layer0, this.weights1), (a) =>
      sigmoid(a)
    );
    this.updatePerceptrons(l1, 1, input_data);
    this.layer2 = math.map(math.multiply(this.layer1, this.weights2), (a) =>
      sigmoid(a)
    );
    this.updatePerceptrons(l2, 2, this.layer1);
    this.layer3 = math.map(math.multiply(this.layer2, this.weights3), (a) =>
      sigmoid(a)
    );
    this.updatePerceptrons(l3, 3, this.layer2);
    if (print_output) {
      console.log(this.layer3);
    }
  }

  backprop(expected_output) {
    const layer3_error = expected_output.map((col, i) =>
      col.map((output_value, j) => output_value - this.layer3[i][j])
    );

    const layer3_delta = calcDelta(layer3_error, this.layer3);
    const layer2_error = calcError(layer3_delta, this.weights3);
    const layer2_delta = calcDelta(layer2_error, this.layer2);
    const layer1_error = calcError(layer2_delta, this.weights2);
    const layer1_delta = calcDelta(layer1_error, this.layer1);

    const dot3 = math.multiply(math.transpose(this.layer2), layer3_delta);
    const dot2 = math.multiply(math.transpose(this.layer1), layer2_delta);
    const dot1 = math.multiply(math.transpose(this.layer0), layer1_delta);

    this.weights3 = this.weights3.map((col, i) =>
      col.map((weight, j) => weight + dot3[i][j])
    );
    this.weights2 = this.weights2.map((col, i) =>
      col.map((weight, j) => weight + dot2[i][j])
    );
    this.weights1 = this.weights1.map((col, i) =>
      col.map((weight, j) => weight + dot1[i][j])
    );
  }

  updatePerceptrons(layer, depth, values) {
    for (let i = 0; i < layer.length; i++) {
      const value = 200 - values[0][i] * 200;
      // layer[i].style.backgroundColor = getPerceptronColor(values[0][i]);
      layer[i].style.backgroundColor = `rgb(${value}, ${value}, ${value})`
      setLabel(layer[i], `label-l${depth}p${i + 1}`, values[0][i].toFixed(2));
    }
  }

  updateInOuts(input, expected_output) {
    for (let i = 0; i < this.num_inputs; i++) {
      setLabel(l0[i], `input${i + 1}`, input[0][i]);
    }
    for (let i = 0; i < this.num_outputs; i++) {
      setLabel(l4[i], `output${i + 1}`, (10 * this.layer3[0][0]).toFixed(2));
      let error;
      if (expected_output[0][i] == 0) {
        error =
          Math.abs(expected_output[0][i] + 1 - (this.layer3[0][0] + 1)) /
          (expected_output[0][i] + 1);
      } else {
        error =
          Math.abs(expected_output[0][i] - this.layer3[0][0]) /
          expected_output[0][i];
      }
      l4[i].style.backgroundColor = getOutputColor(error * 10);
      this.sum_error += error;
    }
  }

  drawLines() {
    const oldSVG = document.querySelector(".connections");
    oldSVG.remove();
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "connections");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    // const percent_complete =
    //   this.num_iter / (TRAINING_ITERATIONS * this.dataset_length);
    const error_modifier = Math.max(1 - this.sum_error / this.num_iter, 0.001);
    for (let i = 0; i < this.num_outputs; i++) {
      for (let j = 0; j < this.num_hidden_nodes; j++) {
        drawSvgTriangles(
          svg,
          `.l3p${j + 1}`,
          `.l4p${i + 1}`,
          error_modifier,
          nn.weights3[j][i]
        );
      }
    }
    for (let i = 0; i < this.num_hidden_nodes; i++) {
      for (let j = 0; j < this.num_hidden_nodes; j++) {
        drawSvgTriangles(
          svg,
          `.l2p${j + 1}`,
          `.l3p${i + 1}`,
          error_modifier,
          nn.weights2[j][i]
        );
      }
    }
    for (let i = 0; i < this.num_hidden_nodes; i++) {
      for (let j = 0; j < this.num_inputs; j++) {
        drawSvgTriangles(
          svg,
          `.l1p${j + 1}`,
          `.l2p${i + 1}`,
          error_modifier,
          nn.weights1[j][i]
        );
      }
    }
    for (let i = 0; i < 4; i++) {
      drawSvgTriangles(svg, `.l0p${i + 1}`, `.l1p${i + 1}`, 0.5, 1);
    }
    elS.appendChild(svg);
    refreshElement(svg);
  }

  train(input, output) {
    this.feedforward(input, false);
    this.backprop(output);
    this.drawLines();
  }

  altTrain(input, output) {
    input.forEach((data, i) => {
      setTimeout(() => {
        this.feedforward(data, false);
        this.backprop(output[i]);
        this.drawLines();
        this.updateInOuts(data, output[i]);
        this.num_iter += 1;
        setNumIter(this.num_iter);
        setAvgErr(`${((this.sum_error / this.num_iter) * 100).toFixed(2)}%`);
      }, i * UPDATE_INTERVAL);
    });
  }
}

const nn = new NeuralNetwork(4, 1, 5, expected_output2.length);

for (let i = 1; i <= TRAINING_ITERATIONS; i++) {
  setTimeout(() => {
    nn.altTrain(input_data2, expected_output2);
  }, i * UPDATE_INTERVAL);
}
setTimeout(() => {
  nn.feedforward(input_data, true);
  console.log("weights1", nn.weights1);
  console.log("weights2", nn.weights2);
  console.log("weights3", nn.weights3);
}, TRAINING_ITERATIONS * UPDATE_INTERVAL + 500);
