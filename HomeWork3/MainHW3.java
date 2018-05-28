package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.core.Environment;
import weka.core.Instances;

public class MainHW3 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
        final int K = 20;
        final double[] lPValues = {1.0, 2.0, 3.0, Double.POSITIVE_INFINITY};
        final Knn.WeightingScheme[] Weights ={Knn.WeightingScheme.Uniform, Knn.WeightingScheme.Weighted};
	    Knn.WeightingScheme bestWeightScheme;

	    Instances dataSet = loadData("auto_price.txt");
	    dataSet.randomize(new Random());
	    Instances scaledDataSet = new FeatureScaler().scaleData(dataSet);

	    Knn scaledKnn = new Knn();
	    Knn originalKnn = new Knn();
	    scaledKnn.buildClassifier(scaledDataSet);
	    originalKnn.buildClassifier(dataSet);

        int bestOriginalK = 0;
        int bestScaledK = 0;
	    Knn.WeightingScheme bestScaledScheme = null;
	    Knn.WeightingScheme bestOriginalScheme = null;

        double bestOriginalPValue = 0;
        double bestScaledPValue = 0;
        double bestOriginalErrorValue = Double.POSITIVE_INFINITY;
        double bestScaledErrorValue = Double.POSITIVE_INFINITY;
        double currentError;
        double currentScaledError;

        for(int k = 1; k <= K; k++ )
        {
            for(double pValue : lPValues)
            {
                for(Knn.WeightingScheme Weight : Weights)
                {
                    scaledKnn.ChangeKnn(Knn.DistanceCheck.Regular, pValue, k, Weight);
                    originalKnn.ChangeKnn(Knn.DistanceCheck.Regular, pValue, k, Weight);

                    currentError = originalKnn.crossValidationError(dataSet,10);
                    currentScaledError = scaledKnn.crossValidationError(scaledDataSet, 10);

                    if(currentError < bestOriginalErrorValue)
                    {
                        bestOriginalErrorValue = currentError;
                        bestOriginalK = k;
                        bestOriginalPValue = pValue;
                        bestOriginalScheme = Weight;
                    }
                    if(currentScaledError < bestScaledErrorValue)
                    {
                        bestScaledErrorValue = currentError;
                        bestScaledK = k;
                        bestScaledPValue = pValue;
                        bestScaledScheme = Weight;
                    }
                }
            }
        }
        String newLine = System.lineSeparator();
        String lineSeperator = "------------------------------" + newLine;

        System.out.printf("%s Results for original dataset:%s%s " +
                          "Cross validation error with K = %d, lp = %1.1f, majority function = %s " +
                          "for auto_price data is: %f %s"
                ,lineSeperator, newLine, lineSeperator, bestOriginalK,
                bestOriginalPValue, bestOriginalScheme.toString(),bestOriginalErrorValue, newLine);

        System.out.printf("%s Results for scaled dataset:%s%s " +
                        "Cross validation error with K = %d, lp = %1.1f, majority function = %s " +
                        "for auto_price data is: %.3f %s"
                ,lineSeperator,newLine, lineSeperator, bestScaledK,
                bestScaledPValue, bestScaledScheme.toString(),bestScaledErrorValue, newLine);
        //initate new efficient and regular Knn for different checking methods.
        int[] numOfFolds = {dataSet.numInstances(), 50, 10, 5, 3};
        Knn efficientKnn = new Knn();
        efficientKnn.buildClassifier(scaledDataSet);
        efficientKnn.ChangeKnn(Knn.DistanceCheck.Efficient, bestScaledPValue, bestScaledK, bestScaledScheme);
        Knn regularKnn = new Knn();
        regularKnn.buildClassifier(scaledDataSet);
        regularKnn.ChangeKnn(Knn.DistanceCheck.Regular, bestScaledPValue, bestScaledK, bestScaledScheme);

        double efficientCrossValidationError = 0;
        double regularCrossValidationError = 0;
        for(int numOfFold : numOfFolds)
        {
            regularCrossValidationError = regularKnn.crossValidationError(scaledDataSet,numOfFold);
            double regularAverageTime = regularKnn.getM_totalTime()/numOfFold;
            System.out.printf("%s %s Results for %d folds:%s %s" +
                              "Cross validation error of regular knn on auto_price dataset is %.3f" +
                              "and%s the average elapsed time is %f %s" +
                              "The total elapsed time is: %d %s",
            newLine, lineSeperator, numOfFold, newLine, lineSeperator, regularCrossValidationError, newLine,
            regularAverageTime, newLine, regularKnn.getM_totalTime(), newLine);

            System.out.println();

            efficientCrossValidationError = efficientKnn.crossValidationError(scaledDataSet,numOfFold);
            double efficientAverageTime = efficientKnn.getM_totalTime()/numOfFold;
            System.out.printf("%s Results for %d folds:%s %s" +
                            "Cross validation error of efficient knn on auto_price dataset is %.3f" +
                            "and%s the average elapsed time is %f %s" +
                            "The total elapsed time is: %d",
                    lineSeperator, numOfFold, newLine, lineSeperator, efficientCrossValidationError, newLine,efficientAverageTime, newLine, efficientKnn.getM_totalTime());
             System.out.println();
        }
    }
}
