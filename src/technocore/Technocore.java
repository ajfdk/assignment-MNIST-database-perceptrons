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
import java.util.Arrays;

public class Technocore {
	static int EPOCHS = 6;
	static double[][] X = { {0, 1, 0, 1},
							{1, 0, 1, 0},
							{0, 0, 1, 1},
							{1, 1, 0, 0}  };

	static double[][] Y = { {0, 1},
					 		{1, 0}, 
							{0, 1},
							{1, 0}
						};

	public static void main(String[] args) {
		System.out.println("- - - - - - - - - - -   Matrix X  - - - - - - - - - - - - - - - ");
		new Matrix(X).MatrixPrint();
		System.out.println("- - - - - - - - - - -   Matrix Y  - - - - - - - - - - - - - - - ");
		new Matrix(Y).MatrixPrint();
		System.out.println("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
		System.out.println("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");

		Network nn = new Network(4, 3, 2);
		
		nn.SGD(X, Y, EPOCHS, 10, 2);

		System.out.println("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
	}

	// makes a one hot vector using Matrix class constructor
	public static Matrix getOneHotVector(int HotNumber) {
		if (HotNumber > 9 || HotNumber < 0)  // configure these later to be problem agnostic
			throw new RuntimeException("Something bad happened and its not my fault.");
		double[][] HoldingVector = new double[10][1];
		HoldingVector[HotNumber][0] = 1;
		Matrix resultingOneHotVector = new Matrix(HoldingVector);
		return resultingOneHotVector;
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

	double learningRate = 10;

	public Network(int inputDIM, int hiddenDIM, int outputDIM) {
		this.inputDIM = inputDIM;
		this.hiddenDIM = hiddenDIM;
		this.outputDIM = outputDIM;
		/*
		weightsHiddenLayer = new Matrix(hiddenDIM, inputDIM);
		weightsOutputLayer = new Matrix(outputDIM, hiddenDIM);
		biasesH   = new Matrix(hiddenDIM, 1);
		biasesOut = new Matrix(outputDIM, 1);
		*/
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
	public void printWB() {
		System.out.println("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
		System.out.println();
		System.out.printf("biases: Layer 1 (hidden)%n");
		this.biasesH.MatrixPrint();
		System.out.println();

		System.out.printf("biases: Layer 2 (output)%n");
		this.biasesOut.MatrixPrint();
		System.out.println();

		System.out.printf("weights: Layer 1 (hidden)%n");
		this.weightsHiddenLayer.MatrixPrint();
		System.out.println();

		System.out.printf("weights: Layer 2 (output))%n");
		this.weightsOutputLayer.MatrixPrint();
		System.out.println();
	}

	public Matrix feedForward(double[] X) {
		Matrix inputActivation = Matrix.makeMatrixFromArray(X);
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

		for (int j = 0; j < epochs; j++) {
			System.out.printf("~~~~~~~		The Epoch is: %d		~~~~~~~%n", j + 1);
			// randomize minibatches for part 2
			
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
			updateOnMiniBatch(X1, Y1, learningRate, mbSize);
			System.out.println("~~~~");
			printWB();

			// minibatch 2
			b++;
			System.out.println("- - - - - - - - - - - - - - - ");
			System.out.printf("The minibatch number is: %d%n", b);
			System.out.println("~~~~");
			updateOnMiniBatch(X2, Y2, learningRate, mbSize);
			System.out.println("~~~~");
			printWB();
			b = 0;
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

		Matrix outputActivation = new Matrix(2,1);
		for (int i = 0; i < X.length; i++) { // or i < mbSize 
			// do back prop, layers: input > hidden > output
			// feeding forwards
			Matrix inputActivation = Matrix.makeMatrixFromArray(X[i]);
			Matrix hiddenZ = Matrix.MatrixMultiply(weightsHiddenLayer, inputActivation);
			hiddenZ.add(biasesH);
			Matrix hiddenActivation = new Matrix(hiddenZ);
			hiddenActivation = hiddenActivation.sigmoid();
			Matrix outputZ = Matrix.MatrixMultiply(weightsOutputLayer, hiddenActivation);
			outputZ.add(biasesOut);
			outputActivation = new Matrix(outputZ);
			outputActivation = outputActivation.sigmoid();
			
			System.out.println();
			System.out.println("Training Case Activations:");
			outputActivation.MatrixPrint();
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
		// ( random gives a double from 0 to 1 ) times 2 - 1 scales it from -1 to 1 
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
		this(A.data);	}


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
	
	// print a matrix
	public void MatrixPrint() {
		for (int i = 0; i < Rows; i++) {
			for (int j = 0; j < Columns; j++)
				System.out.printf("% -6.6f\t", data[i][j]);
			System.out.println();
		}
	}
}