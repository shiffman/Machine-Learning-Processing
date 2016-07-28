import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

Instances training;
//MultilayerPerceptron mlp;
IBk ibk;
ArrayList<Attribute> attributes;

void setup() {
  size(600, 600);

  try {
    // Three attributes
    Attribute xa = new Attribute("x");
    Attribute ya = new Attribute("y");
    Attribute ca = new Attribute("class");
    attributes = new ArrayList<Attribute>();
    attributes.add(xa);
    attributes.add(ya);
    attributes.add(ca);
    // Make an empty training set
    training = new Instances("what goes here", attributes, 1000);

    // The last element is the "class"?
    training.setClassIndex(2);

    for (int i = 0; i < 10000; i++) {
      // Add training data
      Instance inst = new DenseInstance(3); 
      float x = random(width);
      float y = random(height);
      inst.setValue(xa, x/width); 
      inst.setValue(ya, y/height);     
      // Left or right side of the screen
      float d = dist(x, y, width/2, height/2);
      if (d > 200) {
        inst.setValue(ca, 0);
      } else {
        inst.setValue(ca, 1);
      }
      training.add(inst);
    }

    ibk = new IBk();
    ibk.buildClassifier(training);
  } 
  catch (Exception e) {
    e.printStackTrace();
  }

  background(0);
}

void draw() {

  // Try to classify 10 points at a time.
  for (int i = 0; i < 10; i++) {
    float x = random(width);
    float y = random(height);
    Instance inst = new DenseInstance(3);     
    inst.setValue(training.attribute(0), x/width); 
    inst.setValue(training.attribute(1), y/height);       
  
    
    // "instance" has to be associated with "Instances"
    Instances testData = new Instances("Test Data", attributes, 0);
    testData.add(inst);
    testData.setClassIndex(2);        

    float classification = -1;
    try {
      // have to get the data out of Instances
      classification = (float) ibk.classifyInstance(testData.firstInstance());
      println(classification);
    } 
    catch (Exception e) {
      e.printStackTrace();
    }
    noStroke();
    if (classification > 0.5) {
      fill(0, 0, 255);
    } else {
      fill(0, 255, 0);
    }
    ellipse(x, y, 8, 8);
  }
  //noLoop();
}