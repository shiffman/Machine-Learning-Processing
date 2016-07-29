
Instances training;
ArrayList<Attribute> attributes;
int total;
IBk ibk;
float classification = -1;
void initTraining() {
  int cols = wid / rez;
  int rows = height / rez;

  attributes = new ArrayList<Attribute>();

  for (int i = 0; i <  cols*rows; i++) {
    attributes.add(new Attribute(""+i));
  }
  total = attributes.size();
  println(total);

  attributes.add(new Attribute("jump"));
  training = new Instances("pixels", attributes, 100);
  training.setClassIndex(total-1);
  ibk = new IBk();
}

void train() {

  Instance inst = new DenseInstance(total+1);
  for (int i = 0; i < total; i++) {
    inst.setValue(attributes.get(i), brightness(img.pixels[i]) / 255);
  }

  if (jumping && reinforcement || !jumping && !reinforcement) {
    inst.setValue(attributes.get(total), 1);
  } else {
    inst.setValue(attributes.get(total), 0);
  }

  training.add(inst);
  try {
    ibk.buildClassifier(training);
  } 
  catch (Exception e) {
    e.printStackTrace();
  }
}

boolean guess() {
  Instance inst = new DenseInstance(total+1);
  for (int i = 0; i < total; i++) {
    inst.setValue(attributes.get(i), brightness(img.pixels[i]) / 255);
  }
  //printArray(inst);
  Instances testData = new Instances("Test Data", attributes, 0);
  testData.add(inst);
  testData.setClassIndex(total);
  try {
    // have to get the data out of Instances
    classification = (float) ibk.classifyInstance(testData.firstInstance());
    //println(classification);
  } 
  catch (Exception e) {
    e.printStackTrace();
  }

  if (classification > 0.5) {
    return true;
  } else {
    return false;
  }
}