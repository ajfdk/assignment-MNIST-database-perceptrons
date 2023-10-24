/**
 * Alexander Frederick
 * SID: 103 91 653
 * OCT 8, 2023
 * Assignment 2
 * Technocore is a neural network used to recognize had written digits in the MNIST data set.
 * Program can train a network, read weights from a file
 * print accuracy of a model and save weights and biases to a file
 * using ==> jdk-17.0.8.7
 */
package technocore;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;

public class Technocore {
	static int EPOCHS = 30;
	static double trainingRate = 3.0;
	static double[][] Xpart1 = 
						  { {0, 1, 0, 1},
							{1, 0, 1, 0},
							{0, 0, 1, 1},
							{1, 1, 0, 0}  };

	static double[][] Ypart1 = 
						  { {0, 1},
					 		{1, 0}, 
							{0, 1},
							{1, 0}
						};
	static double[][] X;
	static double[][] Y;
	// ez labels is a easy to use 1d array of the labeled data
	static int[] ezLabels;
	// labels count is a count of each class of digit by index = digit
	static int[] labelsCount;
	static double scalingFactorImage = 1.0/255.0;

	public static void main(String[] args) {
		boolean NETWORK_HAS_STATE = false;

		// load mnist data here
		// works on windows
		String fileNTrain = "data\\mnist_train.csv";
		ArrayList<double[]> dataArray = new ArrayList<>();
		ArrayList<int[]> labelArray = new ArrayList<>();
		try {
			File file = new File(fileNTrain);
			Scanner scanner = new Scanner(file);
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				String[] strValues = line.split(",");

				// label? parse as int
				int[] label = new int[1];
				label[0] = Integer.parseInt(strValues[0]);
				labelArray.add(label);

				double[] fDataArray = new double[strValues.length - 1];
				for (int i = 1; i < fDataArray.length; i++) {
					fDataArray[i - 1] = Double.parseDouble(strValues[i]);
				}
				dataArray.add(fDataArray);
			}
			scanner.close();
		} catch (FileNotFoundException e) {
			System.err.println("File not found: " + e.getMessage());
		}
		// convert form Arraylist to 2d array
		double[][] mnistMainTrain = new double[dataArray.size()][];
		for (int i = 0; i < dataArray.size(); i++) {
			mnistMainTrain[i] = dataArray.get(i);
		}
		// System.out.println(Arrays.toString(mnistMainTrain[0]));

		int[][] mnistLabel = new int[labelArray.size()][];
		for (int i = 0; i < labelArray.size(); i++) {
			mnistLabel[i] = labelArray.get(i);
		}

		ezLabels = new int[labelArray.size()];
		for (int i = 0; i < labelArray.size(); i++) {
			int[] tempy = labelArray.get(i);
			ezLabels[i] = tempy[0];
		}
		labelsCount = new int[10];
		for (int i = 0; i < ezLabels.length; i++) {
			labelsCount[ezLabels[i]]++;
		}
		// System.out.println(Arrays.toString(labelsCount));
		
		// System.out.println(Arrays.toString(ezLabels));
		// System.out.println("ezlabel length");
		// System.out.println(ezLabels.length);

		double[][] labels = OHV(mnistLabel);

		// test
		// for (int i = 500; i < 515; i++) {
		// printArt(mnistMainTrain[i]);
		// System.out.println(Arrays.toString(labels[i]));
		// }
		// System.out.println("- - - - - - - - - - -   Matrix X  - - - - - - - - - - - - - - - ");

		// new Matrix(mnistMainTrain).dimPrint();
		// System.out.println("- - - - - - - - - - -   Matrix Y  - - - - - - - - - - - - - - - ");
		// new Matrix(labels).dimPrint();
		// System.out.println("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
		// System.out.println("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");

		// Network starts here
		// our X and Y
		// X = mnistMainTrain;
		// Y = labels;
		// new Matrix(X).dimPrint();
		// new Matrix(Y).dimPrint();
		Network fatMan = new Network(784, 30, 10, false);
		fatMan.printWB();
		// cls();
		// if (true)
		// 	return;
		try (Scanner scanner = new Scanner(System.in)) {
			// USER control starts here
			int onlyOnce = 0;
			while (onlyOnce < 1) {
			    System.out.println("Choose an option:");
			    System.out.println("[1] Train the network");
			    System.out.println("[2] Load a pre-trained network");
			    if(NETWORK_HAS_STATE) {
					System.out.println("[3] Display network accuracy on TRAINING data");
					System.out.println("[4] Display network accuracy on TESTING data");
					System.out.println("[5] Run network on TESTING data showing images and labels");
					System.out.println("[6] Display the misclassified TESTING images");
					System.out.println("[7] Save the network state to file");
				}
			    System.out.println("[0] Exit");

			    int choice = scanner.nextInt();

			    switch (choice) {
			        case 1:
						// cls();
						fatMan.SGD(mnistMainTrain, labels, EPOCHS, 3, 10);
						fatMan.printWB();
						NETWORK_HAS_STATE = true;

			            break;
			        case 2:
						cls();
			            // Implement loading a pre-trained network logic
			            break;
			        case 3:
						cls();
						if (!NETWORK_HAS_STATE) {
							System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
							System.out.println("I'm sorry, Dave. I'm afraid I can't do that.");
							System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
							break;
						}
			            // Implement displaying network accuracy on TRAINING data logic
			            break;
			        case 4:
						cls();
						if (!NETWORK_HAS_STATE) {
							System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
							System.out.println("I'm sorry, Dave. I'm afraid I can't do that.");
							System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
							break;
						}
			            // Implement displaying network accuracy on TESTING data logic
			            break;
			        case 5:
						cls();
						if (!NETWORK_HAS_STATE) {
							System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
							System.out.println("I'm sorry, Dave. I'm afraid I can't do that.");
							System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
							break;
						}
			            // Implement running network on TESTING data and displaying images and labels logic
			            break;
			        case 6:
						cls();
						if (!NETWORK_HAS_STATE) {
							System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
							System.out.println("I'm sorry, Dave. I'm afraid I can't do that.");
							System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
							break;
						}
			            // Implement displaying misclassified TESTING images logic
			            break;
			        case 7:
						cls();
						if (!NETWORK_HAS_STATE) {
							System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
							System.out.println("I'm sorry, Dave. I'm afraid I can't do that.");
							System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
							break;
						}
			            // Implement saving the network state to a file logic
			            break;
					case 11:
						cls();
						System.out.println("That's ridiculous. It's not even funny.");
						System.out.println("this is the debug command. testing part 1:");
						System.out.println("- - - - - - - - - - -   Matrix X  - - - - - - - - - - - - - - - ");
						new Matrix(Xpart1).printM();
						new Matrix(Xpart1).dimPrint();
						System.out.println("- - - - - - - - - - -   Matrix Y  - - - - - - - - - - - - - - - ");
						new Matrix(Ypart1).printM();
						new Matrix(Ypart1).dimPrint();
						System.out.println("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
						System.out.println("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");

						Network nn = new Network(
							4, 3, 2, true);
						
						nn.SGD(Xpart1, Ypart1, 6, 10.0, 2);

						System.out.println("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
						break;
			        case 0:
						cls();
			            System.out.println("Dave, this conversation can serve no purpose anymore. Good-bye.");
			            System.exit(0);
			        default:
						cls();
						System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
						System.out.println("This mission is too important for me to allow you to jeopardize it.");
						System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
						
			    }
				onlyOnce++;
			}

		}
	}

	// makes a one hot vector using Matrix class constructor
	public static Matrix getOneHotVector(int HotNumber) {
		if (HotNumber > 9 || HotNumber < 0)
			throw new RuntimeException("Something bad happened and its not my fault.");
		double[][] HoldingVector = new double[10][1];
		HoldingVector[HotNumber][0] = 1;
		Matrix resultingOneHotVector = new Matrix(HoldingVector);
		return resultingOneHotVector;
	}
	public static double[][] OHV(int[][] mnistLabel) {
		double[][] vectors = new double[mnistLabel.length][10];
		for (int i = 0; i < mnistLabel.length; i++) {
			if (mnistLabel[i][0] > 9 || mnistLabel[i][0] < 0)
				throw new RuntimeException("Something bad happened and its not my fault.");
			int theINDEXsays = (int)mnistLabel[i][0];
			vectors[i][theINDEXsays] = 1.0;
			// System.out.println((double)mnistLabel[i][0]);
		}
		return vectors;
	}
	public static void cls(){
		// some ANSI escape codes to clear screen
		System.out.print("\033[H\033[2J");  
		System.out.flush();
	}

	public static void printArt(double[] image) {
		if (image.length != 784)
			return;
		int dim = 28;
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				char pixel = characterArt(image[28 * i + j]);
				System.out.print(' ');
				System.out.print(pixel);
			}
			System.out.println();
		}
	}

	public static char characterArt(double rainbow) {
		char[] chars = {' ', '.', ':', 'o', 'O', 'X', 'M', '#', '$', '&', 'Y', '8', '%', 'B', '@'};
		int chard = (int)(rainbow / 17.0);
		if (chard >= 14)
			chard = 14;
		if (chard < 0)
			chard = 0;
		return chars[chard];
	}
	// the Fisher–Yates shuffle from wikipedia
	static void shuffleMe(int[] arrayMe) {
		Random randomMe = ThreadLocalRandom.current();
		for (int i = arrayMe.length - 1; i > 0; i--) {
			int indexMe = randomMe.nextInt(i + 1);

			int aMe = arrayMe[indexMe];
			arrayMe[indexMe] = arrayMe[i];
			arrayMe[i] = aMe;
		}
	}
	static int giveMeIndexOfMax(double[] array) {
		int maxIndex = 0;
		double maxValue = array[0];
		for (int i = 1; i < array.length; i++) {
			if (array[i] >maxValue) {
				maxIndex = i;
				maxValue = array[i];
			}
		}
		return maxIndex;
	}
}

class Network {
	Matrix weightsHiddenLayer; 
	Matrix weightsOutputLayer;
	Matrix biasesH; 
	Matrix biasesOut;

	int inputDIM;
	int hiddenDIM;
	int outputDIM;

	// double learningRate = 10.0;
	private boolean istest;

	public Network(int inputDIM, int hiddenDIM, int outputDIM, boolean istest) {
		this.inputDIM = inputDIM;
		this.hiddenDIM = hiddenDIM;
		this.outputDIM = outputDIM;
		this.istest = istest;

		if(istest) {
			double[][] testingHiddenLayerWeightsA = {	
									{-0.21,  0.72, -0.25,  1.00},
									{-0.94, -0.41, -0.47,  0.63},
									{ 0.15,  0.55, -0.49, -0.75}	};
	
			double[][] testingOutputLayerWeightsA = {	
									{0.76, 0.48, -0.73},
									{0.34, 0.89, -0.23}		};
	
			double[][] testingHiddenLayerBiasesA = {
									{ 0.1 },
									{-0.36},
									{-0.31}					};
			
			double[][] testingOutputLayerBiasesA = {	
									{ 0.16},
									{-0.46}					};
		weightsHiddenLayer = new Matrix(testingHiddenLayerWeightsA);
		weightsOutputLayer = new Matrix(testingOutputLayerWeightsA);
		biasesH 		   = new Matrix(testingHiddenLayerBiasesA);
		biasesOut		   = new Matrix(testingOutputLayerBiasesA);
		}
		else {
		weightsHiddenLayer = new Matrix(hiddenDIM, inputDIM);
		weightsOutputLayer = new Matrix(outputDIM, hiddenDIM);
		biasesH   = new Matrix(hiddenDIM, 1);
		biasesOut = new Matrix(outputDIM, 1);
		}
	}
	public void printWB() {
		try {
			PrintStream consoleOut = System.out;
			PrintStream fileOut = new PrintStream(new FileOutputStream("home.txt"));
			System.setOut(fileOut);

			System.out.println("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
			System.out.println();
			System.out.printf("biases: Layer 1 (hidden)%n");
			this.biasesH.printM();
			// this.biasesH.dimPrint();
			System.out.println();
	
			System.out.printf("biases: Layer 2 (output)%n");
			this.biasesOut.printM();
			// this.biasesOut.dimPrint();
			System.out.println();
	
			System.out.printf("weights: Layer 1 (hidden)%n");
			this.weightsHiddenLayer.printM();
			// this.weightsHiddenLayer.dimPrint();
			System.out.println();
	
			System.out.printf("weights: Layer 2 (output))%n");
			this.weightsOutputLayer.printM();
			// this.weightsOutputLayer.dimPrint();
			System.out.println();

			System.setOut(consoleOut);
		} catch (FileNotFoundException e) {
		}
	}

	
	public int predict(double[] sample) {
		Matrix outputActivationMatrix = feedForward(sample);
		double[] outputActivationVector = Matrix.matrixToVector(outputActivationMatrix);
		return Technocore.giveMeIndexOfMax(outputActivationVector);
	}
	
	
	public Matrix feedForward(double[] X) {
		Matrix inputActivation = Matrix.makeMatrixFromArray(X);
		inputActivation.scale(Technocore.scalingFactorImage);
		Matrix hiddenZ = Matrix.MatrixMultiply(weightsHiddenLayer, inputActivation);
		hiddenZ.add(biasesH);
		Matrix hiddenActivation = new Matrix(hiddenZ);
		hiddenActivation = hiddenActivation.sigmoid();
		Matrix outputZ = Matrix.MatrixMultiply(weightsOutputLayer, hiddenActivation);
		outputZ.add(biasesOut);
		Matrix outputActivation = new Matrix(outputZ);
		outputActivation = outputActivation.sigmoid();
		return outputActivation;
	}
	
	/**
	 * SGD is the stochastic gradient descent method. Calls updateOnMiniBatch.
	 * @param X is the input matrix
	 * @param Y is the classification
	 * @param epochs
	 * @param tRate Training rate
	 * @param mbSize mini batch size
	 */
	public void SGD(double[][] X, double[][] Y, int epochs, double tRate, int mbSize) {
		int b = 0; // batch counter
		int[] shuffledA = new int[X.length];
		for (int i = 0; i < shuffledA.length; i++)
			shuffledA[i] = i;

		for (int j = 0; j < epochs; j++) {
			System.out.printf("~~~~~~~		The Epoch is: %d		~~~~~~~%n", j + 1);
			// randomize minibatches for part 2
			// System.out.println(X[0].length);
			Technocore.shuffleMe(shuffledA);
			// System.out.println(Arrays.toString(shuffledA));
			
			// System.out.println(X.length / mbSize);
			if(!istest) {
				
				// for (int miniB = 0; miniB < (X.length / mbSize); miniB++) {
				for (int miniB = 0; miniB < (X.length / mbSize); miniB++) {
					int startIndex = miniB * mbSize;
					// int endIndex = (miniB + 1) * mbSize;
					
					double[][] X1 = new double[mbSize][this.inputDIM];
					double[][] Y1 = new double[mbSize][this.outputDIM];
					for (int i = 0; i < mbSize; i++) {
						X1[i] = Arrays.copyOfRange(X[shuffledA[startIndex + i]], 0, inputDIM);
						Y1[i] = Arrays.copyOfRange(Y[shuffledA[startIndex + i]], 0, outputDIM);
					}
					// System.out.println(Arrays.toString(X1[0]));

					// System.out.println("~~~~~~~~~~~");
					// for (int i = 0; i < X1.length; i++) {
					// 	Technocore.printArt(X1[i]);
					// 	System.out.printf(" Image label: %d%n", Technocore.giveMeIndexOfMax(Y1[i]));
					// }

					// System.out.println(Arrays.toString(Y1[0]));
					// System.out.println("~~~~~~~~~~~");

					updateOnMiniBatch(X1, Y1, tRate, mbSize);
					
					// System.out.println("- - - - - - - - - - - - - - - ");
					// System.out.printf("The minibatch number is: %d%n", miniB);
					// System.out.println("~~~~");
					// System.out.println("~~~~");
					// printWB();
				}
				// System.out.print("...");
			}
			else {
				// divide up minibatches for test input
				double[][] X1 = new double[2][4];
				double[][] Y1 = new double[2][2];
				double[][] X2 = new double[2][4];
				double[][] Y2 = new double[2][2];
				for (int i = 0; i < 2; i++) {
					X1[i] = Arrays.copyOfRange(X[i], 0, 4);
					Y1[i] = Arrays.copyOfRange(Y[i], 0, 2);
				}
				for (int i = 0; i < 2; i++) {
					X2[i] = Arrays.copyOfRange(X[i+2], 0, 4);
					Y2[i] = Arrays.copyOfRange(Y[i+2], 0, 2);
				}
	
				// minibatch 1
				b++;
				System.out.println("- - - - - - - - - - - - - - - ");
				System.out.printf("The minibatch number is: %d%n", b);
				System.out.println("~~~~");
				updateOnMiniBatch(X1, Y1, tRate, mbSize);
				System.out.println("~~~~");
				// printWB();
	
				// minibatch 2
				b++;
				System.out.println("- - - - - - - - - - - - - - - ");
				System.out.printf("The minibatch number is: %d%n", b);
				System.out.println("~~~~");
				updateOnMiniBatch(X2, Y2, tRate, mbSize);
				System.out.println("~~~~");
				// printWB();
				b = 0;
			}
			// after each epoch print accuracy
			if(!istest) {
				int[] correctPredictions = new int[10];
				int totalSamples = Technocore.ezLabels.length;
				int[] Yb = Technocore.ezLabels;
	
				for (int i = 0; i < totalSamples; i++) {
					int predictedNumber = predict(X[i]);
					if (predictedNumber == Yb[i]) {
						correctPredictions[Yb[i]]++;
					}
				}
				int totalCorrect = 0;
				System.out.println("Epoch accuracy.");
				for (int classLabel = 0; classLabel < 10; classLabel++) {
					System.out.println(Integer.toString(classLabel) + " = " + correctPredictions[classLabel] + "/" + Integer.toString(Technocore.labelsCount[classLabel]));
					totalCorrect += correctPredictions[classLabel];
				}
				double accuracy = (double)(totalCorrect * 100.0 / totalSamples);
				System.out.print("Accuracy = " + Integer.toString(totalCorrect) + "/60000 = ");
				System.out.printf("%.3f", accuracy);
				System.out.println();
				// System.out.printf("%f", accuracy);
			}
		}
	}
	/**
	 * Executes feedforward and backprop formula for sigmoid neurons.
	 * @param X is input
	 * @param Y is input classifier
	 * @param lRate
	 * @param mbSize
	 */
	public void updateOnMiniBatch(double [][] X, double[][] Y, double lRate, int mbSize) {
		// Gradients (nabla) for weights and biases of the hidden and output layers
		Matrix nablaWHid = new Matrix(weightsHiddenLayer);	nablaWHid.zed();
		Matrix nablaWOut = new Matrix(weightsOutputLayer);	nablaWOut.zed();
		Matrix nablaBHid = new Matrix(biasesH);				nablaBHid.zed();
		Matrix nablaBOut = new Matrix(biasesOut);			nablaBOut.zed();

		Matrix outputActivation = new Matrix(this.outputDIM,1);
		for (int i = 0; i < X.length; i++) { // or i < mbSize 
			// do back prop, layers: input > hidden > output
			// feeding forwards
			Matrix inputActivation = Matrix.makeMatrixFromArray(X[i]);
			// scale divide by 255
			// inputActivation.printM();
			inputActivation.scale(Technocore.scalingFactorImage);
			// System.out.println("AFTER SCALING");
			// inputActivation.printM();

			Matrix hiddenZ = Matrix.MatrixMultiply(weightsHiddenLayer, inputActivation);
			hiddenZ.add(biasesH);
			Matrix hiddenActivation = new Matrix(hiddenZ);
			hiddenActivation = hiddenActivation.sigmoid();
			Matrix outputZ = Matrix.MatrixMultiply(weightsOutputLayer, hiddenActivation);
			outputZ.add(biasesOut);
			outputActivation = new Matrix(outputZ);
			outputActivation = outputActivation.sigmoid();
			
			// System.out.println();
			// System.out.println("Training Case Activations:");
			// outputActivation.printM();
			// System.out.println("Which number highest?:");
			// double[] prediction = Matrix.matrixToVector(outputActivation);
			// outputActivation.dimPrint();
			// System.out.printf("Which number highest?: %d%n", Technocore.giveMeIndexOfMax(prediction));
			
			// ff done
			
			// backwards passthrough
			Matrix yMatrix = Matrix.makeMatrixFromArray(Y[i]);
			Matrix deltanablaBOut = Matrix.subtract(outputActivation, yMatrix); // cost fn
			deltanablaBOut.hadamardProduct(outputZ.derivativeSigmoid());
			nablaBOut.add(deltanablaBOut);
			
			Matrix deltanablaWOut = Matrix.MatrixMultiply(deltanablaBOut, Matrix.transpose(hiddenActivation));
			nablaWOut.add(deltanablaWOut);

			Matrix deltanablaBHid = Matrix.MatrixMultiply(Matrix.transpose(weightsOutputLayer),deltanablaBOut);
			deltanablaBHid.hadamardProduct(hiddenZ.derivativeSigmoid());
			nablaBHid.add(deltanablaBHid);
			
			Matrix deltanablaWHid = Matrix.MatrixMultiply(deltanablaBHid, Matrix.transpose(inputActivation));
			nablaWHid.add(deltanablaWHid);
		}
		// done with mini batch. update weights and biases
		nablaWOut.scale(lRate/mbSize);
		this.weightsOutputLayer = Matrix.subtract(this.weightsOutputLayer, nablaWOut);
		nablaWHid.scale(lRate/mbSize);
		this.weightsHiddenLayer = Matrix.subtract(this.weightsHiddenLayer, nablaWHid);
		nablaBOut.scale(lRate/mbSize);
		this.biasesOut = Matrix.subtract(this.biasesOut, nablaBOut);
		nablaBHid.scale(lRate/mbSize);
		this.biasesH = Matrix.subtract(this.biasesH, nablaBHid);
	}

}

/**
 * from my linear algebra textbook. a matrix is defined by the size of its rows and columns.
 * m rows and n columns 
*/
class Matrix {
	private int Rows; 	  // M rows
	private int Columns;  // N columns
	private double[][] data;
	
	// constructor for m ROWS × n COLUMNS matrix
	public Matrix(int Rows, int Columns) {
		this.Rows = Rows;
		this.Columns = Columns;
		data = new double[Rows][Columns];

		// random numbers from -1 to 1
		// we can randomize the matrix on construction
		// for (int i = 0; i < Rows; i++)
		// 	for ( int j = 0; j < Columns; j++)
		// 		this.data[i][j] = Math.random() * 2 - 1;
		// note: ( random gives a double from 0 to 1 ) times 2 - 1 scales it from -1 to 1 
		Random Randy = new Random();
		double mean = 0;
		double stdDev = 0.01;
		for (int i = 0; i < Rows; i++)
			for ( int j = 0; j < Columns; j++)
				this.data[i][j] = Randy.nextGaussian() * stdDev + mean;
	}

	// constructor for passing 2 d array to the matrix class
	public Matrix(double[][] data) {
		Rows = data.length;
		Columns = data[0].length;
		this.data = new double[Rows][Columns];
		// copy array into matrix data
		for (int i = 0; i < Rows; i++)
			for ( int j = 0; j < Columns; j++)
				this.data[i][j] = data[i][j];
	}
	public static Matrix makeMatrixFromArray(double[] A) {
		// constructing a matrix from a 1 d array
		Matrix R = new Matrix(A.length, 1);
		for (int i = 0; i < A.length; i++) 
			R.data[i][0] = A[i];
		return R;
	}
	
	// copy a matrix object
	public Matrix(Matrix A) {
		this(A.data);
	}

	public static double[][] matrixToArray(Matrix A) {
		return A.data;
	}
	public static double[] matrixToVector(Matrix A) {
		if (A.data[0].length == 1) {
			double[] returnMe = new double[A.data.length];
			for (int i = 0; i < A.data.length; i++) {
				returnMe[i] = A.data[i][0];
			}
			return returnMe;
		}
		else {
			return null;
		}
	}

	/**
	 * Matrix Multiplication,  return matrix R = AD , note: this is not a commutative operation
	 * @param A Left hand Matrix
	 * @param D Right hand Matrix
	 * @return Matrix Product
	 */
	public static Matrix MatrixMultiply(Matrix A, Matrix D) {
		if (A.Columns != D.Rows)
			throw new RuntimeException("My textbook says not to do that...");
		Matrix R = new Matrix(A.Rows, D.Columns);
		// for this you just look at your text book and read off the summation thingy
		for (int i = 0; i < R.Rows; i++)
			for (int j = 0; j < R.Columns; j++)
				for (int k = 0; k < A.Columns; k++)
					R.data[i][j] += (A.data[i][k] * D.data[k][j]);
		return R;
	}

	/**
	 * a element wise multiply. also called Hadamard product.
	 * changes this matrix
	 */
	public void hadamardProduct(Matrix D){

		if (D.Rows != this.Rows || D.Columns != this.Columns)
			throw new RuntimeException("You can't element wise multiply matrices that way!");
		for (int i = 0; i < Rows; i++)
			for (int j = 0; j < Columns; j++)
				this.data[i][j] = this.data[i][j] * D.data[i][j];
	}

	/**
	 * hadamard product -> new matrix
	 * @param A
	 * @param D
	 * @return
	 */
	public static Matrix hProd(Matrix A, Matrix D) {
		if (D.Rows != A.Rows || D.Columns != A.Columns)
			throw new RuntimeException("You can't element wise multiply matrices that way!");
		Matrix R = new Matrix(A.Rows, A.Columns);
		for (int i = 0; i < A.Rows; i++)
			for (int j = 0; j < A.Columns; j++)
				R.data[i][j] = A.data[i][j] * D.data[i][j];
		return R;
	}

	/**
	 * scale modifiers this matrix, entry wise
	 */
	public void scale(double scalingValue) {
		for (int i = 0; i < Rows; i++)
			for (int j = 0; j < Columns; j++)
				this.data[i][j] = this.data[i][j] * scalingValue;
	}

	/**
	 * zerorizes your matrix
	 */
	public void zed() {
		for (int i = 0; i < Rows; i++)
			for (int j = 0; j < Columns; j++)
				this.data[i][j] = 0;
	}

	public void add(double scalar) {
		for (int i = 0; i < Rows; i++)
			for (int j = 0; j < Columns; j++)
				this.data[i][j] = this.data[i][j] + scalar;
	}

	public void add(Matrix D) {
		if (D.Rows != this.Rows || D.Columns != this.Columns)
			throw new RuntimeException("You can't add matrices that way!");
		for (int i = 0; i < Rows; i++)
			for (int j = 0; j < Columns; j++)
				this.data[i][j] = this.data[i][j] + D.data[i][j];
	}

	public static Matrix subtract(Matrix A, Matrix D) {
		if (D.Rows != A.Rows || D.Columns != A.Columns)
			throw new RuntimeException("You can't subtract matrices that way!");
		Matrix R = new Matrix(A.Rows, A.Columns);
		for (int i = 0; i < A.Rows; i++)
			for (int j = 0; j < A.Columns; j++)
				R.data[i][j] = A.data[i][j] - D.data[i][j];
		return R;
	}

	/**
	 * transpose fn for turning a column vector into a row vector and the reverse. 
	 * transpose mirrors a matrix across the diagonal.
	 * @param D
	 * @return transpose of Matrix D
	 */
	public static Matrix transpose(Matrix D) {
		// note the row and col are swapped
		Matrix TransposedMatrix = new Matrix(D.Columns, D.Rows);
		for (int i = 0; i < D.Rows; i++)
			for (int j = 0; j < D.Columns; j++)
				TransposedMatrix.data[j][i] = D.data[i][j];
		return TransposedMatrix;
	}

	// spread sheet uses e = 2.71828
	public Matrix sigmoid() {
		Matrix R = new Matrix(this.Rows, this.Columns);
		for (int i = 0; i < this.Rows; i++) {
			for (int j = 0; j < this.Columns; j++)
				R.data[i][j] = 1 / (1 + Math.pow(2.71828, -this.data[i][j]));
		}
		return R;
	}

	// sigmoid fn given double
	public double s(double z) {
		return (1 / (1 + Math.pow(2.71828, - z)));
	}

	// also we can use a method to apply the derivative of the sigmoid to each element of an array
	public Matrix derivativeSigmoid() {
		Matrix R = new Matrix(this.Rows, this.Columns);
		for (int i = 0; i < Rows; i++) {
			for (int j = 0; j < Columns; j++) {
				double z = this.data[i][j];
				// from the derivative of the sigmoid or logistic function
				// this is the a * ( 1 - a ) part in the gradient formula
				R.data[i][j] = s(z) * (1 - s(z));
			}
		}
		return R;
	}
	public void dimPrint() {
		System.out.printf("Dimensionality: %d by %d %n", this.Rows, this.Columns);
	}

	public void printM() {
		for (int i = 0; i < Rows; i++) {
			for (int j = 0; j < Columns; j++)
				// System.out.printf("% -2.f\t", data[i][j]);
				System.out.printf("% -6.6f\t", data[i][j]);
			System.out.println();
		}
	}
}