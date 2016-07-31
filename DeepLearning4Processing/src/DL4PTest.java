import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import processing.core.PApplet;


/**
 * "Linear" Data Classification Example
 * 
 * Modified for use with Processing by Daniel Shiffman
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots) (Now replaced with Processing)
 *
 */
public class DL4PTest extends PApplet { 
	public static void main(String[] args) {
		PApplet.main(new String[] {"DL4PTest"});
	}

	MultiLayerNetwork model;

	public void settings() {
		size(400, 400);

	}

	public void setup() {
		int seed = 123;
		double learningRate = 0.1;
		int nEpochs = 500;

		int numInputs = 2;
		int numOutputs = 2;
		int numHiddenNodes = 20;


		background(127);

		int rows = 2500;

		float[][] dataf = new float[rows][2];
		float[][] datal = new float[rows][2];

		for (int i = 0; i <  rows; i++) {
			float x = random(width);
			float y = random(height);
			float d = dist(x,y,width/2,height/2);

			dataf[i][0] = x / width;
			dataf[i][1] = y / height;

			//if (x > width/2) {
			if (d > 100) {
				fill(0);
				datal[i][0] = 1;
				datal[i][1] = 0;
			} else {
				fill(255);
				datal[i][0] = 0;
				datal[i][1] = 1;
			}
			noStroke();
			ellipse(x, y, 8, 8);
		}

		INDArray features = Nd4j.create(dataf);
		INDArray labels = Nd4j.create(datal);

		DataSet ds = new DataSet(features,labels);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		.seed(seed)
		.iterations(1)
		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		.learningRate(learningRate)
		.updater(Updater.NESTEROVS).momentum(0.9)
		.list()
		.layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
				.weightInit(WeightInit.XAVIER)
				.activation("relu")
				.build())
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
				.weightInit(WeightInit.XAVIER)
				.activation("softmax").weightInit(WeightInit.XAVIER)
				.nIn(numHiddenNodes).nOut(numOutputs).build())
				.pretrain(false).backprop(true).build();


		model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(10));


		for ( int n = 0; n < nEpochs; n++) {
			model.fit(ds);
		}
		noStroke();
		fill(0,0,255,100);
		rect(0,0,width,height);
	}

	public void draw() {
		//background(0);
		for (int i = 0; i < 25; i++) {
			float x = random(width);
			float y = random(height);

			float[] features = new float[2];
			features[0] = x/width;
			features[1] = y/height;

			INDArray predicted = model.output(Nd4j.create(features),false);
			float z = predicted.getFloat(0);

			noStroke();
			if (z < 0.5) {
				fill(255, 0, 0);				
			} else {
				fill(0, 255, 0);
			}
			ellipse(x,y,8,8);
		}


	}
}
