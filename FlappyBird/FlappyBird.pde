Bird b;
ArrayList<Pipe> pipes = new ArrayList<Pipe>();

float score = 0;

void setup() {
  size(400, 600);
  b = new Bird();
  pixelDensity(2);
}

void draw() {
  background(0);

  if (frameCount % 200 == 0) {
    pipes.add(new Pipe());
  }

  if (keyPressed) {
    PVector up = new PVector(0, -0.75);
    b.applyForce(up);
  }
  b.update();
  b.show();

  for (int i = pipes.size()-1; i>=0; i--) {
    Pipe p = pipes.get(i);
    p.update();


    if (p.hits(b)) {
      p.show(true);
    } else {
      p.show(false);
    }


    if (p.x < -p.w) {
      pipes.remove(i);
    }
  }
}