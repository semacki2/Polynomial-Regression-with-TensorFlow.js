let xs = [];
let ys = [];
let a, b, c, d;

// create an optimizer to minimize our loss
// we are using stochasitc gradient descent
// slowling tweaking the variable of our polynomial regression function
// until it accuratly predicts y values.

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(400, 400);
  background(0);

  createP(
    "Click on the canvas to add points. </br> Refresh the page to start over."
  );

  //these need to be variables because we will be changing them
  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
  d = tf.variable(tf.scalar(random(-1, 1)));
}

function draw() {
  //if there are values in our arrays
  if (xs.length > 1 && ys.length > 1) {
    //use tf.tidy() to prevent memory leaks
    tf.tidy(() => {
      //create a tensor out of the outputs
      const tfys = tf.tensor1d(ys);

      //train the model
      //adjust m and b to minimize the "loss" function
      //the minimize function automatically trains by adjusting variables.
      //the only variables we created were m and b.
      //we could be explicit and define a varList of variables to adjust.
      optimizer.minimize(() => loss(predict(xs), tfys));
    });

  }

  background(0);

  stroke(255);
  strokeWeight(8);

  for (let i = 0; i < xs.length; i++) {
    let px = map(xs[i], -1, 1, 0, width);
    let py = map(ys[i], 1, -1, 0, height);
    point(px, py);
  }

  //draw the graph if there are 2 or more points
  if (xs.length > 1 && ys.length > 1) {
    const curveX = [];
    for (let i = -1; i <= 1; i += 0.01) {
      curveX.push(i);
    }

    //tys = tensor y values
    const tys = tf.tidy(() => predict(curveX));

    //convert it from a tensor back to an array
    let curveY = tys.dataSync();

    tys.dispose();

    beginShape();
    noFill();
    stroke(255);
    strokeWeight(1);
    for (let i = 0; i < curveX.length; i++) {
      let x = map(curveX[i], -1, 1, 0, width);
      let y = map(curveY[i], -1, 1, height, 0);
      vertex(x, y);
    }
    endShape();
  }
}

function mouseDragged() {
  let x = map(mouseX, 0, width, -1, 1);
  let y = map(mouseY, 0, height, 1, -1);

  xs.push(x);
  ys.push(y);
}

//function that takes in an array of x values and returns an array of y values.
function predict(xs) {
  //create a tensor out of the inputs
  const tfxs = tf.tensor1d(xs);

  //y = ax^3 + bx^2 + cx + d
  //in this case, a, b and c are global variables.
  const ys = tfxs
    .pow(tf.scalar(3))
    .mul(a)
    .add(tfxs.square().mul(b))
    .add(tfxs.mul(c))
    .add(d);

  return ys;
}

//loss function. tells us how wrong our model was.
//predictions are the y values from the predict function
//labels are the "true" y values stored in the ys array
//finds the distance between each prediction and the "true" value.
//squares the difference.
//returns the mean of all the squared errors.
function loss(predictions, ys) {
  return predictions.sub(ys).square().mean();
}
