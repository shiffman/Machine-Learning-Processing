import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

import processing.core.PApplet;




/**
 * From: http://deeplearning4j.org/mnist-for-beginners.html
 * http://deeplearning4j.org/simple-image-load-transform
 * 
 * Modified for use with Processing by Daniel Shiffman
 * *
 */
public class DL4P_ImageClassifier extends PApplet { 
	public static void main(String[] args) {
		PApplet.main(new String[] {"DL4P_ImageClassifier"});
	}



	int imgH = 28;
	int imgW = 28;
	int channels = 3;
	int outputNum = 3;
	int numExamples = 80;


	//int numRows = 28; // The number of rows of a matrix.
	//int numColumns = 28; // The number of columns of a matrix.
	// int outputNum = 10; // Number of possible outcomes (e.g. labels 0 through 9). 
	int batchSize = 128; // How many examples to fetch with each step. 
	int rngSeed = 123; // This random-number generator applies a seed to ensure that the same initial weights are used when training. We’ll explain why this matters later. 
	int numEpochs = 15; // An epoch is a complete pass through a given dataset. 


	Random rand = new Random(rngSeed);
	//	String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;


	public void settings() {
		size(400, 400);
	}

	public void setup() {
		//		MnistDataSetIterator mnistTrain = null;
		//		MnistDataSetIterator mnistTest = null;
		//		try {
		//			mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
		//			mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
		//		} catch (IOException e) {
		//			// TODO Auto-generated catch block
		//			e.printStackTrace();
		//		}

		File dir = new File(dataPath(""));
		String[] ext = {"jpg"};
		FileSplit filesInDir = new FileSplit(dir, ext, rand);
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		BalancedPathFilter pathFilter = new BalancedPathFilter(rand, ext, labelMaker);

		InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
		InputSplit trainData = filesInDirSplit[0];
		InputSplit testData = filesInDirSplit[1];

		ImageRecordReader recordReader = new ImageRecordReader(imgH, imgW, channels, labelMaker);
		ImageTransform transform = new MultiImageTransform(rand);
		try {
			recordReader.initialize(trainData,transform);
		} catch (IOException e) {
			e.printStackTrace();
		}

		//convert the record reader to an iterator for training - Refer to other examples for how to use an iterator
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);


		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		.seed(rngSeed)
		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		.iterations(1)
		.learningRate(0.006)
		.updater(Updater.NESTEROVS).momentum(0.9)
		.regularization(true).l2(1e-4)
		.list()
		.layer(0, new DenseLayer.Builder()
		.nIn(imgH * imgW * channels) // Number of input datapoints.
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
		// Additional step to make this work with pairing 4D loaded images with CNN.
		// From @AlexBlack
		// ok, got it. so the ImageRecordReader outputs data in 4d array format 
		// suitable for CNNs (convolutional, subsampling layers)
		// whereas dense layer expects 2d format
		// you can add a preprocessor to convert between the two
		.inputPreProcessor(0, new CnnToFeedForwardPreProcessor(imgH,imgW,channels))
		.build();


		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(1));

		println("Train model....");
		for( int i=0; i<numEpochs; i++ ){
			model.fit(dataIter);
		}
		dataIter.reset();


		println("Evaluate model....");
		Evaluation eval = new Evaluation(outputNum);
		while(dataIter.hasNext()){
			DataSet next = dataIter.next();
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
