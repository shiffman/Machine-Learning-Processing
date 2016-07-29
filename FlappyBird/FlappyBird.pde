Bird b;
ArrayList<Pipe> pipes = new ArrayList<Pipe>();

float score = 0;

int wid = 400;

PImage img;

int rez = 20;

boolean reinforcement = false;
boolean jumping = false;


void setup() {
  size(800, 400);
  b = new Bird();
  //pixelDensity(2);
  pipes.add(new Pipe());
  img = createImage(wid/rez, height/rez, RGB);
  smooth();

  initTraining();
}

void draw() {
  background(0);

  if (frameCount % 200 == 0) {
    pipes.add(new Pipe());
  }

  /*if (keyPressed) {
   PVector up = new PVector(0, -0.75);
   b.applyForce(up);
   }*/
  train();

  if (guess()) {
    PVector up = new PVector(0, -0.75);
    b.applyForce(up);
    jumping = true;
  } else {
    jumping = false;
  }


  b.update();
  b.show();

  boolean safe = true;
  
  for (int i = pipes.size()-1; i>=0; i--) {
    Pipe p = pipes.get(i);
    p.update();


    if (p.hits(b)) {
      p.show(true);
      safe = false;
      reinforcement = false;
    } else {
      reinforcement = true;
      p.show(false);
    }

    if (p.x < -p.w) {
      pipes.remove(i);
    }
  }
  
  if (safe) {
    score+=0.01;
  } else {
    score-=0.1;
  }
  score = constrain(score, 0, 1);


  loadPixels();
  img.loadPixels();
  for (int x = 0; x < img.width; x++) {
    for (int y = 0; y < img.height; y++) {
      float sum = 0;
      for (int i = 0; i < rez; i++) {
        for (int j = 0; j < rez; j++) {
          int index = (x * rez + i) + (y * rez + j) * width;
          float b = brightness(pixels[index]);
          sum += b;
        }
      }
      sum /= rez*rez;
      img.pixels[x + y * img.width] = color(sum);
    }
  }
  fill(127);
  stroke(127);
  rect(wid, 0, wid, height);
  img.updatePixels();
  imageMode(CENTER);
  image(img, wid + wid/2, height/2, wid/2, height/2);
  
  fill(0);
  textSize(24);
  textAlign(CENTER);
  if (jumping) {
    text("JUMP : " + nf(classification, 1,2), wid + wid/2, height - 50);
  } else {
    text("DON'T JUMP : " + nf(classification, 1,2), wid + wid/2, height - 50);
  }
    text("SCORE : " + nf(score, 1,2), wid + wid/2, height - 25);


}