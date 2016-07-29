class Pipe {
  float x;
  float top;
  float bottom;
  float w = 40;

  Pipe() {
    x = width + w;
    top = random(height/2);
    bottom = random(height/2);
  }

  boolean hits(Bird b) {
    if (b.pos.x > x && b.pos.x < x + w) {
      if ((b.pos.y < top+b.r) || (b.pos.y > (height-bottom-b.r))) {
        return true;
      }
    }
    return false;
  }

  void update() {
    x--;
  }

  void show(boolean hit) {
    stroke(255);
    if (hit) {      
      fill(255, 0, 0);
    } else {
      fill(255);
    }
    rect(x, 0, w, top); 
    rect(x, height-bottom, w, bottom);
  }
}