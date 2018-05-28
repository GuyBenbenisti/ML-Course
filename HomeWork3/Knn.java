package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Comparator;

class DistanceCalculator {

    private double m_pValue;
    private Knn.DistanceCheck m_typeOfCheck;
    private double m_threshold;

    DistanceCalculator(double i_pValue, Knn.DistanceCheck i_typeOfCheck, double i_threshold) {
        m_pValue = i_pValue;
        m_typeOfCheck = i_typeOfCheck;
        m_threshold = i_threshold;
    }

    /**
     * We leave it up to you wheter you want the distance method to get all relevant
     * parameters(lp, efficient, etc..) or have it has a class variables.
     */
    double distance(Instance one, Instance two) {
        double distance = 0;
        switch (m_typeOfCheck) {
            case Regular:
                if (this.m_pValue == Double.POSITIVE_INFINITY) {
                    distance = lInfinityDistance(one, two);
                    break;
                }
                distance = lpDistance(one, two);
                break;
            case Efficient:
                if (this.m_pValue == Double.POSITIVE_INFINITY) {
                    distance = efficientLInfinityDistance(one, two);
                    break;
                }
                distance = efficientLpDistance(one, two);
                break;
        }
        return distance;
    }

    /**
     * Returns the Lp distance between 2 instances.
     *
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two) {
        double sum = 0;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            sum += Math.pow(Math.abs(one.value(i) - two.value(i)), this.m_pValue);
        }
        sum = Math.pow(sum, 1.0 / m_pValue);
        return sum;
    }

    /**
     * Returns the L infinity distance between 2 instances.
     *
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        double maxValue = Double.MIN_VALUE;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            maxValue = Math.max(maxValue, Math.abs((one.value(i) - two.value(i))));
        }
        return maxValue;
    }


    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     *
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDistance(Instance one, Instance two) {
        double sum = 0;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            sum += Math.pow(Math.abs(one.value(i) - two.value(i)), m_pValue);
            if (sum >= m_threshold) {
                return -1;
            }
        }
        return Math.pow(sum, 1.0 / m_pValue);
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     *
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two) {
        double maxValue = Double.MIN_VALUE;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            maxValue = Math.max(maxValue, Math.abs((one.value(i) - two.value(i))));
            if (maxValue >= m_threshold) {
                return -1;
            }
        }
        return maxValue;
    }

    /**
     * the following are gets and sets methods for the class values.
     */
    public double getpValue() {
        return m_pValue;
    }

    public void setPvalue(double i_Value) {
        m_pValue = i_Value;
    }

    public Knn.DistanceCheck getTypeOfCheck() {
        return m_typeOfCheck;
    }

    public void setTypeOfCheck(Knn.DistanceCheck typeOfCheck) {
        this.m_typeOfCheck = typeOfCheck;
    }

    public double getM_threshold() {
        return m_threshold;
    }

    void setM_threshold(double m_threshold) {
        this.m_threshold = m_threshold;
    }
}

/**
 * This class is exactly like the instance, besides having an extra paramter keeping distance value.
 */
class InstanceD implements Comparable<InstanceD> {
    private Instance m_Instace;
    private double m_Distance;

    InstanceD(Instance i_Instance, double i_Distance) {
        m_Instace = i_Instance;
        m_Distance = i_Distance;
    }

    Instance getInstance() {
        return m_Instace;
    }

    double getDistance() {
        return m_Distance;
    }

    @Override
    public int compareTo(InstanceD two) {
        if (m_Distance - two.getDistance() < 0) {
            return -1;
        } else if (m_Distance - two.getDistance() > 0) {
            return 1;
        }

        return 0;
    }
}

public class Knn implements Classifier {

    public enum DistanceCheck {Regular, Efficient}

    public enum WeightingScheme {Uniform, Weighted}

    private Instances m_trainingInstances;
    private DistanceCheck m_typeOfCheck;
    private double m_pValue;
    private int m_kNN;
    private WeightingScheme m_WeightScheme;

    public long getM_totalTime() {
        return m_totalTime;
    }

    private long m_totalTime;

    public void ChangeKnn(DistanceCheck m_typeOfCheck, double m_pValue, int m_kNN, WeightingScheme m_WeightScheme) {
        this.m_typeOfCheck = m_typeOfCheck;
        this.m_pValue = m_pValue;
        this.m_kNN = m_kNN;
        this.m_WeightScheme = m_WeightScheme;
    }

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
        m_trainingInstances = instances;
    }

    /**
     * Returns the knn prediction on the given instance.
     *
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        InstanceD[] kNN = findNearestNeighbors(instance);
        double returnValue = 0;
        switch (m_WeightScheme) {
            case Uniform:
                returnValue = getAverageValue(kNN);
                break;
            case Weighted:
                returnValue = getWeightedAverageValue(kNN);
                break;
        }
        return returnValue;
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     *
     * @param instances
     * @return
     */
    public double calcAvgError(Instances instances) {
        double predictedValue;
        double targetValue;
        double numOfErrors = 0;

        for (Instance curInstance : instances) {
            predictedValue = regressionPrediction(curInstance);
            targetValue = curInstance.classValue();
            numOfErrors += Math.abs(predictedValue - targetValue);
        }
        return numOfErrors / instances.numInstances();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     *
     * @param instances     Insances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds) {
        long totalTime = 0;
        long startTime;
        double crossValidationError = 0;
        Instances testInstaces;

        for(int i = 0; i < num_of_folds; i ++)
        {
            testInstaces = instances.testCV(num_of_folds, i);
            m_trainingInstances = instances.trainCV(num_of_folds, i);
            startTime = System.nanoTime();
            crossValidationError += calcAvgError(testInstaces);
            totalTime += (System.nanoTime() - startTime);
        }
        m_totalTime = totalTime;
        return crossValidationError / num_of_folds;
    }


    /**
     * Finds the k nearest neighbors.
     *
     * @param instance
     */
    public InstanceD[] findNearestNeighbors(Instance instance) {
        InstanceD[] kNN = initiateKNN(m_kNN);
        int maxDisInKnnIndex;
        double curDistance;
        DistanceCalculator distanceCalculator = new DistanceCalculator(m_pValue, m_typeOfCheck, Double.MAX_VALUE);

        for (Instance curInstance : m_trainingInstances) {
            switch (m_typeOfCheck) {
                case Regular:
                    if (!curInstance.equals(instance)) {
                        curDistance = distanceCalculator.distance(instance, curInstance);
                        if (curDistance < kNN[m_kNN - 1].getDistance()) {
                            kNN[m_kNN - 1] = new InstanceD(curInstance, curDistance);
                            Arrays.sort(kNN);
                        }
                    }
                    break;
                case Efficient:
                    if (!curInstance.equals(instance)) {
                        curDistance = distanceCalculator.distance(instance, curInstance);
                        if (curDistance != -1) {
                            if (curDistance < kNN[m_kNN - 1].getDistance()) {
                                kNN[m_kNN - 1] = new InstanceD(curInstance, curDistance);
                                Arrays.sort(kNN);
                                distanceCalculator.setM_threshold(Math.pow(kNN[m_kNN - 1].getDistance(), m_pValue));
                            }
                        }
                    }
                    break;
            }
        }
        return kNN;
    }

    private InstanceD[] initiateKNN(int i_size) {
        InstanceD[] output = new InstanceD[i_size];
        for (int i = 0; i < output.length; i ++){
            output[i] = new InstanceD(m_trainingInstances.firstInstance(), Double.POSITIVE_INFINITY);
        }
        return output;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     *
     * @param
     * @return
     */
    public double getAverageValue(InstanceD[] i_KnnInstances) {
        double averageValue = 0;
        for (int i = 0; i < i_KnnInstances.length; i++) {
            averageValue += i_KnnInstances[i].getInstance().classValue();
        }
        return averageValue / i_KnnInstances.length;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     *
     * @return
     */
    public double getWeightedAverageValue(InstanceD[] i_KnnInstances) {
        double averageValue = 0;
        double sumWeights = 0;
        double curWi;

        for (InstanceD curInstanceD : i_KnnInstances) {
            if (curInstanceD.getDistance() == 0) {
                return curInstanceD.getInstance().classValue();
            }
            curWi = 1.0 / Math.pow(curInstanceD.getDistance(), 2.0);
            sumWeights += curWi;
            averageValue += curWi * curInstanceD.getInstance().classValue();
        }
        return averageValue / sumWeights;
    }


    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }
}
