package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW2 {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    /**
     * Sets the class index as the last attribute.
     * @param fileName
     * @return Instances data
     * @throws IOException
     */
    public static Instances loadData(String fileName) throws IOException{
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {
        Instances trainingCancer = loadData("cancer_train.txt");
        Instances testingCancer = loadData("cancer_test.txt");
        Instances validationCancer = loadData("cancer_validation.txt");
        double[] pValues = {1.0, 0.75, 0.5, 0.25, 0.05, 0.005 };
        //Choosing an impurity measure.
        DecisionTree giniTree = new DecisionTree("Gini", 1);
        DecisionTree entropyTree = new DecisionTree("Entropy", 1);
        giniTree.buildClassifier(trainingCancer);
        entropyTree.buildClassifier(trainingCancer);
        double entropyValidationError = entropyTree.calcAvgError(validationCancer);
        double giniValidationError = giniTree.calcAvgError(validationCancer);
        String impurityType = entropyValidationError < giniValidationError ? "Entropy" : "Gini";

        System.out.println("Validation error using Entropy Method: " + entropyTree.calcAvgError(validationCancer));
        System.out.println("Validation error using Gini Method: " + giniTree.calcAvgError(validationCancer));
        //Creating pValuesTrees and finding the best validation error and the best tree;
        DecisionTree[] pValuesTrees = new DecisionTree[pValues.length];
        int bestTreeIndex = -1;
        double bestError = Double.MAX_VALUE;
        double curError,trainingError;
        for (int i = 0; i < pValuesTrees.length; i++)
        {
            pValuesTrees[i] = new DecisionTree(impurityType, pValues[i]);
            pValuesTrees[i].buildClassifier(trainingCancer);
            pValuesTrees[i].calculateHeights(validationCancer);
            trainingError =pValuesTrees[i].calcAvgError(trainingCancer);
            curError = pValuesTrees[i].calcAvgError(validationCancer);
            //Printing the information.
            System.out.printf("----------------------------------------------------\n"
                            + "Decision Tree with p_value of: %1.3f\n"
                            + "The train error of the decision tree is %1.4f\n"
                            + "Max height on validation data: %d\n"
                            + "Average height on validation data: %1.2f\n"
                            + "The Validation error of the decision tree is %1.4f\n",
                    pValues[i],
                    trainingError,
                    pValuesTrees[i].getMaxHeight(),
                    pValuesTrees[i].getAvgHeight(),
                    curError);
            if(bestError > curError)
            {
                bestError = curError;
                bestTreeIndex = i;
            }
        }
        DecisionTree bestTree = pValuesTrees[bestTreeIndex];
        System.out.println("----------------------------------------------------\n");
        System.out.printf("Best validation error at p_value = %1.3f\n"
                        + "Test error with best tree: %1.5f\n"
                        + "----------------------------------------------------\n",
                pValues[bestTreeIndex],
                bestTree.calcAvgError(testingCancer));

        //bestTree.printTree();
        bestTree.printTreeTest(trainingCancer);
    }
}
