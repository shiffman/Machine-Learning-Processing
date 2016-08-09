import java.io.IOException;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
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
 * From: http://deeplearning4j.org/mnist-for-beginners.html
 * 
 * Modified for use with Processing by Daniel Shiffman
 * *
 */
public class DL4P_MNIST extends PApplet { 
	public static void main(String[] args) {
		PApplet.main(new String[] {"DL4P_MNIST"});
	}

	int numRows = 28; // The number of rows of a matrix.
	int numColumns = 28; // The number of columns of a matrix.
	int outputNum = 10; // Number of possible outcomes (e.g. labels 0 through 9). 
	int batchSize = 128; // How many examples to fetch with each step. 
	int rngSeed = 123; // This random-number generator applies a seed to ensure that the same initial weights are used when training. We’ll explain why this matters later. 
	int numEpochs = 15; // An epoch is a complete pass through a given dataset. 


	public void settings() {

		size(400, 400);

	}

	public void setup() {
		MnistDataSetIterator mnistTrain = null;
		MnistDataSetIterator mnistTest = null;
		try {
			mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
			mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		.seed(rngSeed)
		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		.iterations(1)
		.learningRate(0.006)
		.updater(Updater.NESTEROVS).momentum(0.9)
		.regularization(true).l2(1e-4)
		.list()
		.layer(0, new DenseLayer.Builder()
		.nIn(numRows * numColumns) // Number of input datapoints.
		.nOut(1000) // Number of output datapoints.
		.activation("relu") // Activation function.
		.weightInit(WeightInit.XAVIER) // Weight initialization.
		.build())
		.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
		.nIn(1000)
		.nOut(outputNum)
		.activation("softmax")
		.weightInit(WeightInit.XAVIER)
		.build())
		.pretrain(false).backprop(true)
		.build();


		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(1));

		println("Train model....");
		for( int i=0; i<numEpochs; i++ ){
			model.fit(mnistTrain);
		}


		println("Evaluate model....");
		Evaluation eval = new Evaluation(outputNum);
		while(mnistTest.hasNext()){
			DataSet next = mnistTest.next();
			INDArray output = model.output(next.getFeatureMatrix());
			eval.eval(next.getLabels(), output);
		}

		println(eval.stats());
		println("****************Example finished********************");
	}

	public void draw() {
		noLoop();
	}



}
