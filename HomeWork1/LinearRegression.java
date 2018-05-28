package HomeWork1;

import java.util.Arrays;

import javax.xml.crypto.Data;

import org.omg.CORBA.Current;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {

    private int m_ClassIndex;

    private int m_truNumAttributes;

    private double[] m_coefficients;

    private double m_alpha;

    // the method which runs to train the linear regression predictor, i.e.
    // finds its weights.
    @Override
    public void buildClassifier(Instances trainingData) throws Exception {

        m_ClassIndex = trainingData.classIndex();

        m_truNumAttributes = trainingData.numAttributes() - 1;

        m_coefficients = new double[m_truNumAttributes + 1];

        findAlpha(trainingData);

        for (int l = 0; l < m_coefficients.length; l++) {
            // reset the m_coefficients
            m_coefficients[l] = 1;
        }

        double current_error = Double.MAX_VALUE;

        double previous_error = 0;

        while (Math.abs(current_error - previous_error) > 0.003) {

            previous_error = current_error;

            for (int i = 0; i < 100; i++) {

                m_coefficients = gradientDescent(trainingData);
            }

            current_error = calculateMSE(trainingData);
        }

    }

    // the method which runs to train the linear regression predictor, i.e.
    // finds its weights.
    // Just like buildClassifier but uses the given alpha.
    public void buildClassifier(Instances Data, double alpha) throws Exception {

        m_ClassIndex = Data.classIndex();

        m_truNumAttributes = Data.numAttributes() - 1;

        m_coefficients = new double[m_truNumAttributes + 1];

        m_alpha = alpha;

        for (int l = 0; l < m_coefficients.length; l++) {
            // reset the m_coefficients
            m_coefficients[l] = 1;
        }
        double current_error = Double.MAX_VALUE;

        double previous_error = 0;

        while (Math.abs(current_error - previous_error) > 0.003) {

            previous_error = current_error;

            for (int i = 0; i < 100; i++) {

                m_coefficients = gradientDescent(Data);
            }
            current_error = calculateMSE(Data);
        }

    }

    private void findAlpha(Instances data) throws Exception {

        double min_error = Double.MAX_VALUE;

        double min_alpha = Double.MAX_VALUE;
        // initialize minimum error and alpha in the beginning to be maximum value
        double current_error;

        double previous_error;

        for (int i = -17; i <= 0; i++) {

            previous_error = Double.MAX_VALUE;

            current_error = 0;

            m_alpha = Math.pow(3.0, i);

            for (int l = 0; l < m_coefficients.length; l++) {
                // reset the m_coefficients
                m_coefficients[l] = 1;
            }

            current_error = calculateError(data, current_error, previous_error);

            if (min_error > current_error) {

                min_alpha = m_alpha;

                min_error = current_error;
            }
        }
        m_alpha = min_alpha;

    }

    private double calculateError(Instances data, double current, double previous) throws Exception {

        double current_error = current;

        double previous_error = previous;

        for (int j = 1; j < 20000; j++) {

            m_coefficients = gradientDescent(data);

            if (j % 100 == 0) {

                current_error = calculateMSE(data);

                if (current_error > previous_error) {

                    return previous_error;

                } else {

                    previous_error = current_error;
                }
            }
        }
        return current_error;
    }

    /**
     * An implementation of the gradient descent algorithm which should return the
     * weights of a linear regression predictor which minimizes the average squared
     * error.
     *
     * @param trainingData
     * @throws Exception
     */
    private double[] gradientDescent(Instances trainingData) throws Exception {

        double[] temps = Arrays.copyOf(m_coefficients, m_truNumAttributes + 1);

        for (int i = 0; i < temps.length; i++)
        // update coefficients in the temps array for all Theta Indexes.
        {
            // using the gradient descent method learned in class.
            temps[i] = m_coefficients[i] - (m_alpha * partial_derivatives(i, trainingData));
        }

        // copy the updated coefficients into m_coefficients
        for (int j = 0; j < m_coefficients.length; j++) {

            m_coefficients[j] = temps[j];
        }
        return m_coefficients;
    }

    /**
     * The function calculates the partial derivatives with respect to 0i
     *
     * @param i
     * @param data
     * @return partial derivatives
     * @throws Exception
     */
    private double partial_derivatives(int i, Instances data) throws Exception {

        double partial_derivative = 0;

        if (i == 0)
        // if index i is 0 we don't multiply
        {
            for (int j = 0; j < data.numInstances(); j++) {

                double h0i = regressionPrediction(data.instance(j));

                double y0i = data.instance(j).value(m_ClassIndex);

                partial_derivative += (h0i - y0i); // calculating (h0(Xi) - Yi)

            }
            return (partial_derivative / data.numInstances());
        }
        // if index i is not 0.
        for (int j = 0; j < data.numInstances(); j++) {

            double h0i = regressionPrediction(data.instance(j));

            double y0i = data.instance(j).value(m_ClassIndex);

            double attribute_i = data.instance(j).value(i - 1);

            partial_derivative += (h0i - y0i) * attribute_i; // calculating (h0(Xi) - Yi) * Xi
        }
        return (partial_derivative / data.numInstances());
    }

    /**
     * Returns the prediction of a linear regression predictor with weights given by
     * m_coefficients on a single instance.
     *
     * @param instance
     * @return
     * @throws Exception
     */
    public double regressionPrediction(Instance instance) throws Exception {

        double inner_product = m_coefficients[0];

        for (int i = 1; i <= m_ClassIndex; i++) {

            inner_product += (m_coefficients[i] * instance.value(i - 1));
        }
        return inner_product;
    }

    /**
     * Calculates the total squared error over the data on a linear regression
     * predictor with weights given by m_coefficients.
     *
     * @param data
     * @return
     * @throws Exception
     */
    public double calculateMSE(Instances data) throws Exception {

        double MSE = 0;

        for (int i = 0; i < data.numInstances(); i++) {

            double h0i = regressionPrediction(data.instance(i));

            double y0i = data.instance(i).value(m_ClassIndex);

            MSE += Math.pow((h0i - y0i), 2.0);
        }

        return (MSE / (2 * data.numInstances()));
    }

    @Override
    public double classifyInstance(Instance arg0) throws Exception {
        // Don't change
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // Don't change
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // Don't change
        return null;
    }

    public double Get_Alpha() {
        return m_alpha;
    }
}
