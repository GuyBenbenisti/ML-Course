package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.DirectoryStream.Filter;
import java.util.concurrent.ThreadLocalRandom;

import weka.classifiers.bayes.net.search.global.K2;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

public class MainHW1 {

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
     *
     * @param fileName
     * @return Instances data
     * @throws IOException
     */
    public static Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {
        // load data
        Instances training_data = loadData(
                "wind_training.txt");

        Instances testing_data = loadData(
                "wind_testing.txt");

        LinearRegression MAL_PREDICTION = new LinearRegression();

        // find best alpha and build classifier wit h all attributes

        MAL_PREDICTION.buildClassifier(training_data);

        double best_alpha = MAL_PREDICTION.Get_Alpha();

        double trainingError = MAL_PREDICTION.calculateMSE(training_data);

        double testingError = MAL_PREDICTION.calculateMSE(testing_data);

        System.out.println("The chosen alpha is: " + best_alpha);

        System.out.println("Training error with all features is: " + trainingError);

        System.out.println("Test error with all features is: " + testingError);

        // build classifiers with all 3 attributes combinations
        int[] best_features = new int[4];

        best_features[3] = training_data.classIndex();

        double current_error = Double.MAX_VALUE;

        double min_error = Double.MAX_VALUE;

        Remove remove = new Remove();

        for (int l = 0; l < training_data.numAttributes() - 1; l++) {
            for (int m = l + 1; m < training_data.numAttributes() - 1; m++) {
                for (int n = m + 1; n < training_data.numAttributes() - 1; n++) {
                    LinearRegression current_run = new LinearRegression();

                    int[] current_features = { l, m, n, training_data.classIndex() };

                    remove.setAttributeIndicesArray(current_features);

                    remove.setInvertSelection(true);
                    // keeps only l,m,n and class index columns in order to train on this data
                    // alone.
                    remove.setInputFormat(training_data);

                    Instances current_attributes = weka.filters.Filter.useFilter(training_data, remove);

                    current_attributes.setClassIndex(current_attributes.numAttributes() - 1);

                    current_run.buildClassifier(current_attributes, best_alpha);

                    current_error = current_run.calculateMSE(current_attributes);

                    System.out.printf("%s|%s|%s \t %2.15f \n", current_attributes.attribute(0).name(),
                            current_attributes.attribute(1).name(), current_attributes.attribute(2).name(),
                            current_error);

                    if (current_error < min_error) {
                        min_error = current_error;

                        best_features[0] = l;

                        best_features[1] = m;

                        best_features[2] = n;

                    }
                }
            }
        }

        System.out.printf("%1s %s|%s|%s : %2.15f \n", "Training error the features",
                training_data.attribute(best_features[0]).name(), training_data.attribute(best_features[1]).name(),
                training_data.attribute(best_features[2]).name(), min_error);
        // Calculating the mean squared error on training data and test data:
        // First sorting the data
        Remove best_remove = new Remove();

        best_remove.setAttributeIndicesArray(best_features);

        best_remove.setInputFormat(training_data);

        best_remove.setInvertSelection(true);

        Instances best_training_attributes = weka.filters.Filter.useFilter(training_data, remove);

        Instances best_testing_attributes = weka.filters.Filter.useFilter(testing_data, remove);

        best_training_attributes.setClassIndex(best_training_attributes.numAttributes() - 1);

        best_testing_attributes.setClassIndex(best_testing_attributes.numAttributes() - 1);

        LinearRegression best_model = new LinearRegression();

        // train the model with the best 3 attributes found before
        best_model.buildClassifier(best_training_attributes);

        System.out.printf("%1s %s|%s|%s : %2.15f \n", "Test error the features",
                testing_data.attribute(best_features[0]).name(), testing_data.attribute(best_features[1]).name(),
                testing_data.attribute(best_features[2]).name(), best_model.calculateMSE(best_testing_attributes));

    }

}
