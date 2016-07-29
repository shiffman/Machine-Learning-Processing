import weka.classifiers.functions.MultilayerPerceptron;
//import weka.classifiers.functions.MLPClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

Instances training;
MLPClassifier mlp;

void setup() {
  size(600, 600);

  try {
    // Three attributes
    Attribute xa = new Attribute("x");
    Attribute ya = new Attribute("y");
    Attribute ca = new Attribute("class");
    ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    attributes.add(xa);
    attributes.add(ya);
    attributes.add(ca);
    // Make an empty training set
    training = new Instances("what goes here", attributes, 1000);

    // The last element is the "class"?
    training.setClassIndex(2);

    for (int i = 0; i < 1000; i++) {
      // Add training data
      Instance inst = new DenseInstance(3); 
      float x = random(width);
      float y = random(height);
      inst.setValue(xa, x/width); 
      inst.setValue(ya, y/height);     
      //// Left or right side of the screen
      //if (x > 300) {
      //  inst.setValue(ca, 0);
      //} else {
      //  inst.setValue(ca, 1);
      //}
      float d = dist(x, y, width/2, height/2);
      if (d > 200) {
        inst.setValue(ca, 0);
      } else {
        inst.setValue(ca, 1);
      }
      
      
      training.add(inst);
    }

    // Try a perceptron
    mlp = new MultilayerPerceptron();
    // These are arbitrary
    
    mlp.setLearningRate(0.5);
    mlp.setMomentum(0.6);
    mlp.setTrainingTime(5000);
    mlp.setHiddenLayers("3");
    mlp.buildClassifier(training);
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
    float classification = -1;
    try {
      classification = (float) mlp.classifyInstance(inst);
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
}