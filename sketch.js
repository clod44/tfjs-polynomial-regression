let x_vals = [];
let y_vals = [];


//ax^3 + bx^2 + cx + d
let a,b,c,d;


let lossVal = 0;
let lossValPrev = 0;
let lossValGraph;
let lossValGraphIndex = 0;
let lossValGraphColor;
let learningRate = 0.1;
let optimizer = tf.train.adam(learningRate);

function setup() {
    const canvas = createCanvas(600, 600);
    canvas.parent('sketch-holder');
    a = tf.variable(tf.scalar(random(2)-1));
    b = tf.variable(tf.scalar(random(2)-1));
    c = tf.variable(tf.scalar(random(2)-1));
    d = tf.variable(tf.scalar(random(2)-1));

    lossValGraph = createGraphics(width, height);
    lossValGraphColor = color(0,0,100);
}

function loss(pred, labels) {
    // just some formula to give you a value that
    // represents the loss/fault rate/value
    return pred.sub(labels).square().mean();
}

function predict(xs) {
    const tfxs = tf.tensor1d(xs);
    //ax^3 + bx^2 + cx + d
    const ys =  tfxs.pow(tf.scalar(3)).mul(a) 
                .add(tfxs.square().mul(b))
                .add(tfxs.mul(c))
                .add(d);

    return ys;
}

function mousePressed() {
    // we are mapping them because its easier to look at
    //check for the boundaries
    if (mouseX > 0 && mouseX < width && mouseY > 0 && mouseY < height) {
        x_vals.push(map(mouseX, 0, width, -1, 1));
        y_vals.push(map(mouseY, 0, height, 1, -1));

        lossValGraph.clear();
        lossValGraphIndex = 0;
        lossValGraphColor = color(random(50)+50,random(50)+50,random(50)+50);
    }
}

function draw() {

    background(0);


    // run the optimizer with our loss function
    // we give it our x values and get a guess y values from loss function
    // then our loss function compares guessed y values to actual y values
    // then loss function returns a loss value
    // optimizer uses this value
    tf.tidy(() => {
        if (x_vals.length > 0) {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(()=>{
                const temp = loss(predict(x_vals), ys);
                lossVal = temp.dataSync();
                return temp;
            });
        }
    });

    



    //draw lossValGraph
    lossValGraph.strokeWeight(1);
    lossValGraph.stroke(0);
    lossValGraph.line(lossValGraphIndex%width, 0, lossValGraphIndex%width, height);
    lossValGraph.stroke(lossValGraphColor);
    lossValGraph.line(lossValGraphIndex%width, height, lossValGraphIndex%width, height-lossVal*10*height);
    lossValGraph.stroke(0);
    lossValGraph.fill(100);
    lossValGraph.textSize(10);
    lossValGraph.text("0.1 loss", 5,10);
    lossValGraph.text("0 loss", 5,height-10);
    lossValGraphIndex++;
    image(lossValGraph,0,0);


    //draw grid
    stroke(255,100);
    strokeWeight(1);
    const gridSize = 50;
    for (let i = 0; i < width; i += gridSize) {
        line(i, 0, i, height);
    }
    for (let i = 0; i < height; i += gridSize) {
        line(0, i, width, i);
    }


    //draw guessed heights of points and offset lines
    stroke(255,0,0);

    const guessYs = tf.tidy(()=>predict(x_vals).dataSync());
    for (let i = 0; i < x_vals.length; i++) {

        const guessY = map(guessYs[i],-1,1,height,0);
        const realX = map(x_vals[i],-1,1,0,width);
        const realY = map(y_vals[i],-1,1,height,0);

        strokeWeight(10);
        point(realX, guessY);
        
        strokeWeight(2);
        line(realX,realY,realX,guessY);
        
    }


    //draw user points
    stroke(255);
    strokeWeight(10);
    for (let i = 0; i < x_vals.length; i++) {
        const px = map(x_vals[i], -1, 1, 0, width);
        const py = map(y_vals[i], -1, 1, height, 0);
        point(px, py);
    }


    // we gonna draw from x=0 to x=1
    const curveX = [];
    for (let x = -1; x < 1.01; x+=0.05) {
        curveX.push(x);
    }

    const ys = tf.tidy(() => predict(curveX));
    const curveY = ys.dataSync();
    ys.dispose();


    noFill();
    stroke(0,255,0);
    strokeWeight(3);
    beginShape();
    for (let i = 0; i < curveX.length; i++) {
        const x = map(curveX[i],-1,1,0,width);
        const y = map(curveY[i],-1,1,height,0);
        vertex(x,y);
    }
    endShape();

    

    stroke(0);
    fill(255);
    strokeWeight(2);
    addText("optimizer: adam");
    addText("learning rate: " + learningRate);
    fill(255,100,0);
    addText("polynominal: ax^3+bx^2+cx+d");
    fill(255);
    addText("a: " + a.dataSync()[0]);
    addText("b: " + b.dataSync()[0]);
    addText("c: " + c.dataSync()[0]);
    addText("d: " + c.dataSync()[0]);
    fill(lossVal > lossValPrev ? color(255,0,0) : color(0,255,0));
    lossValPrev = lossVal;
    addText("loss: " + lossVal);
    fill(100);
    addText("allocated memory at end: " + tf.memory().numBytes+" Bytes");
    addText("tensors: " + tf.memory().numTensors);

    textOffset=20;
}

let textOffset = 20;
function addText(texthere){
    text(texthere, 20, textOffset);
    textOffset+=20;
}

document.getElementById("sldrLearningRate").value = 0.1;
document.getElementById("sldrLearningRate").addEventListener("change",(e)=>{
    learningRate = e.target.value;
    optimizer = tf.train.sgd(learningRate);
});

document.getElementById("btnResetPoints").addEventListener("click",()=>{
    x_vals = [];
    y_vals = [];
    lossVal = 0;
    lossValGraph.background(0);
    tf.tidy(()=>{
        a.assign(tf.scalar(random(2)-1));
        b.assign(tf.scalar(random(2)-1));
        c.assign(tf.scalar(random(2)-1));
        d.assign(tf.scalar(random(2)-1));
    });
});

document.getElementById("btnPause").addEventListener("click",()=>{
    noLoop();
});

document.getElementById("btnResume").addEventListener("click",()=>{
    loop();
});


