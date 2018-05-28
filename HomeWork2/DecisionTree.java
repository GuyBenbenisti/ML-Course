package HomeWork2;

import java.util.Iterator;
import java.util.LinkedList;

import javax.xml.crypto.Data;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;

class Node {
    Node[] children;
    Node parent;
    int attributeIndex;
    double returnValue;

}

enum impurity {
    entropy, gini;

    public static impurity fromString(String str) {
        impurity type = null;
        if (str.equalsIgnoreCase("entropy")) {
            type = entropy;
        } else if (str.equalsIgnoreCase("gini")) {
            type = gini;
        }
        return type;
    }
}

public class DecisionTree implements Classifier {
    private Node rootNode;
    private impurity type;
    private int treeMaxHeight;
    private double treeAvgHeight;
    private double pValue;
    private double[][] chiTable = {
            { 0.102, 0.575, 1.213, 1.923, 2.675, 3.455, 4.255, 5.071, 5.899, 6.737, 7.584, 8.438 }, // p-value 0.75
            { 0.455, 1.386, 2.366, 3.357, 4.351, 5.348, 6.346, 7.344, 8.343, 9.342, 10.341, 11.340 }, // p-value 0.5
            { 1.323, 2.773, 4.108, 5.385, 6.626, 7.841, 9.037, 10.219, 11.389, 12.549, 13.701, 14.845 }, // p-value 0.25
            { 3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026 }, // p-value 0.05
            { 7.879, 10.597, 12.838, 14.860, 16.750, 18.548, 20.278, 21.955, 23.589, 25.188, 26.757, 28.300 }// p-value	0.005
    };
    private double[] pValues = {1, 0.75, 0.5, 0.25, 0.05, 0.005 };
    /**
     * Builds a new decision tree and set the impurity according to the input string.
     * @param str
     */
    public DecisionTree(String str, double pValue) {
        this.type = impurity.fromString(str);
        this.pValue = pValue;
        rootNode = new Node();
        rootNode.parent = null;
        this.treeAvgHeight = 0;
        this.treeMaxHeight = 0;
    }

    @Override
    public double classifyInstance(Instance instance) {
        Node curNode = rootNode;
        Node prevNode = new Node();
        double instanceAttributeValue;
        while(curNode.children != null)
        {
            prevNode = curNode;
            instanceAttributeValue = instance.value(curNode.attributeIndex);
            curNode = curNode.children[(int) instanceAttributeValue];
            if(curNode == null) {
                return prevNode.returnValue;
            }
        }
        return curNode.returnValue;

    }

    @Override
    public void buildClassifier(Instances arg0) throws Exception {
        buildTree(arg0, rootNode);
    }

    /**
     * Calculate the average error on a given instances set.
     * @param i_Data
     * @return
     */
    public double calcAvgError(Instances i_Data)
    {
        double numErrors = 0;
        double currentClassification;
        for(Instance instance : i_Data)
        {
            currentClassification = classifyInstance(instance);
            if(currentClassification != instance.classValue())
            {
                numErrors ++;
            }
        }
        return numErrors / i_Data.numInstances();
    }

    /**
     * Builds the decision tree on given data set using a recursive method
     * @param i_Data
     * @param curNode
     */
    public void buildTree(Instances i_Data, Node curNode) {
        int bestAttributeIndex = findBestSplitAttribute(i_Data);
        //Stopping condition for no more attributes to split by case
        if (bestAttributeIndex == -1) {
            curNode.children = null;
            curNode.attributeIndex = -1;
            curNode.returnValue = calcReturnValue(i_Data);
        }
        //Stopping condition for perfect classification case
        else if (isPerfectClassification(i_Data)) {
            curNode.children = null;
            curNode.attributeIndex = -1;
            curNode.returnValue = calcReturnValue(i_Data);
        }
        else {
            curNode.returnValue = calcReturnValue(i_Data);
            curNode.attributeIndex = -1;
            if(!proneOrNot(i_Data, bestAttributeIndex))
            {
                curNode.attributeIndex = bestAttributeIndex;
                Instances[] sortedByBestAttribute = sortInstancesByAttributeValue(i_Data, bestAttributeIndex);
                int numValues = sortedByBestAttribute.length;
                curNode.children = new Node[numValues];
                for (int i = 0; i < curNode.children.length; i++) {
                    if (sortedByBestAttribute[i].numInstances() == 0) {
                        curNode.children[i] = null;
                    } else {
                        curNode.children[i] = new Node();
                        curNode.children[i].parent = curNode;
                        buildTree(sortedByBestAttribute[i], curNode.children[i]);
                    }
                }
            }
        }
    }

    /**
     * calculates the gain of splitting the input data according to the attribute
     * @param i_Data
     * @param i_AttributeIndex
     * @return
     */
    public double calcGain(Instances i_Data, int i_AttributeIndex)
    {
        double gain = 0;
        double sumV = 0;
        double[] prob = calcProb(i_Data, i_AttributeIndex);
        Instances[] instanceByAttribute = sortInstancesByAttributeValue(i_Data, i_AttributeIndex);
        switch (this.type) {
            case gini:
                double giniIndex = calcGini(calcProb(i_Data, i_Data.classIndex()));
                for (int i = 0; i < instanceByAttribute.length; i++) {
                    double giniIndexSv = calcGini(calcProb(instanceByAttribute[i], instanceByAttribute[i].classIndex()));
                    sumV += prob[i] * giniIndexSv;
                }
                gain = giniIndex - sumV;
                break;
            case entropy:
                double entropy = calcEntropy(calcProb(i_Data, i_Data.classIndex()));
                for (int i = 0; i < instanceByAttribute.length; i++) {
                    double entropySv = calcEntropy(calcProb(instanceByAttribute[i], instanceByAttribute[i].classIndex()));
                    sumV += (prob[i] * entropySv);
                }
                gain = entropy - sumV;
                break;
        }
        return gain;
    }

    /**
     * Calculates the Entropy of a random variable
     *
     * @param i_Prob
     * @return
     */
    public double calcEntropy(double[] i_Prob) {
        double probSum = 0;
        for (int i = 0; i < i_Prob.length; i++) {
            if (i_Prob[i] != 0) {
                probSum += (i_Prob[i] * (Math.log(i_Prob[i]) / Math.log(2.0)));
            } else
                probSum += 0;
        }
        return (0 - probSum);
    }

    /**
     * Calculates the Gini of a random variable
     *
     * @param i_Prob
     * @return
     */
    public double calcGini(double[] i_Prob) {
        double probSum = 0;
        for (int i = 0; i < i_Prob.length; i++) {
            if (i_Prob[i] != 0) {
                probSum += Math.pow(i_Prob[i], 2);
            } else
                probSum += 0;
        }
        return (1 - probSum);
    }



    public double calcChiSquare(Instances i_Data, int i_AttributeIndex) {
        double[] probSet = calcProb(i_Data, i_Data.classIndex());
        double p0 = probSet[0];
        double p1 = probSet[1];
        double ChiSquare = 0;
        Instances[] instanceByAttribute = sortInstancesByAttributeValue(i_Data, i_AttributeIndex);
        double Pf, Nf, Df, E0, E1;
        for (int i = 0; i < instanceByAttribute.length; i++) {
            if (instanceByAttribute[i].numInstances() != 0) {
                probSet = calcProb(instanceByAttribute[i], instanceByAttribute[i].classIndex());
                Df = instanceByAttribute[i].numInstances();
                E0 = Df * p0;
                E1 = Df * p1;
                Pf = probSet[0] * instanceByAttribute[i].numInstances();
                Nf = probSet[1] * instanceByAttribute[i].numInstances();
                if (E0 != 0 && E1 != 0) {
                    ChiSquare += (Math.pow(Pf - E0, 2) / E0) + (Math.pow(Nf - E1, 2) / E1);
                }
            }
        }
        return ChiSquare;
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

    public double getAvgHeight()
    {
        return treeAvgHeight;
    }

    public int getMaxHeight()
    {
        return treeMaxHeight;
    }

    /**
     * updates the tree max Height and avg Height fields
     * @param i_Data
     */
    public void calculateHeights(Instances i_Data){
        int maxHeight = 0;
        double sumOfHeights = 0;
        int curHeight = 0;
        for(Instance instance : i_Data)
        {
            curHeight = Height(instance);
            sumOfHeights += curHeight;
            if(curHeight > maxHeight)
            {
                maxHeight = curHeight;
            }
        }
        this.treeMaxHeight = maxHeight;
        this.treeAvgHeight = sumOfHeights / i_Data.numInstances();
    }

    public void printTree(){
        StringBuilder sb = new StringBuilder();
        sb.append("Root\n");
        sb.append("Returning value: " + this.rootNode.returnValue + "\n");
        printTreeHelper(this.rootNode, sb, 1);
        System.out.println(sb.toString());
    }

    private void printTreeHelper(Node currentNode, StringBuilder sb, int tabs){
        if (currentNode.children == null) {
            addTabs(tabs, sb);
            sb.append(currentNode.children == null ? "Leaf. Returning value: " + currentNode.returnValue + "\n" : "");
            return;
        }

        for (int i = 0; i < currentNode.children.length; i++){

            if (currentNode.children[i] != null){
                addTabs(tabs, sb);
                sb.append("If attribute " + currentNode.attributeIndex + " = " + i + "\n");
                if (currentNode.children[i].children != null){
                    addTabs(tabs, sb);
                    sb.append("Returning value: " + currentNode.children[i].returnValue + "\n");
                }
                printTreeHelper(currentNode.children[i], sb, tabs + 1);
            }
        }
    }

    private void addTabs(int tabs, StringBuilder sb) {
        for (int i = 0; i < tabs; i++)
            sb.append("\t");
    }
    private int degreeOfFreedom(Instances[] i_Data)
    {
        int degreeOfFreedom = 0;
        for (int i = 0; i < i_Data.length; i++) {
            if(i_Data[i].numInstances() != 0)
            {
                degreeOfFreedom++;
            }
        }
        return degreeOfFreedom;
    }

    private boolean isPerfectClassification(Instances i_Data) {
        boolean output = false;
        Instances[] sortedByClass = sortInstancesByAttributeValue(i_Data, i_Data.classIndex());
        if (sortedByClass[0].numInstances() == i_Data.numInstances()
                || sortedByClass[1].numInstances() == i_Data.numInstances()) {
            output = true;
        }
        return output;
    }

    private int calcReturnValue(Instances i_Data) {
        int sumOfZeros = 0;
        int sumOfOnes = 0;
        for (int i = 0; i < i_Data.numInstances(); i++) {
            if (i_Data.instance(i).classValue() == 0) {
                sumOfZeros++;
            } else if (i_Data.instance(i).classValue() == 1) {
                sumOfOnes++;
            }
        }
        return sumOfZeros > sumOfOnes ? 0 : 1;
    }

    private int findBestSplitAttribute(Instances i_Data) {
        int bestAttributeIndex = -1;
        double bestGain = 0;
        double currentGain;
        for (int i = 0; i < i_Data.numAttributes() - 1; i++) {
            currentGain = calcGain(i_Data, i);
            if (currentGain > bestGain) {
                bestAttributeIndex = i;
                bestGain = currentGain;
            }
        }
        return bestAttributeIndex;
    }

    /**
     *
     * @param i_Data
     * @return
     */
    private double[] calcProb(Instances i_Data, int i_AttributeIndex) {
        int AttributeNumValues = i_Data.attribute(i_AttributeIndex).numValues();
        double[] output = new double[AttributeNumValues];

        for (int i = 0; i < i_Data.numInstances(); i++) {
            int indexOfValue = (int) i_Data.instance(i).value(i_AttributeIndex);
            output[indexOfValue]++;
        }
        for (int i = 0; i < AttributeNumValues; i++) {
            if (i_Data.numInstances() != 0) {
                output[i] = (double) output[i] / i_Data.numInstances();
            }
        }
        return output;
    }

    /**
     * Return an array of instances
     *
     * @param i_Data
     * @param i_AttributeIndex
     * @return
     */
    private Instances[] sortInstancesByAttributeValue(Instances i_Data, int i_AttributeIndex) {
        int AttributeNumValues = i_Data.attribute(i_AttributeIndex).numValues();
        Instances[] output = new Instances[AttributeNumValues];
        for (int i = 0; i < AttributeNumValues; i++) {
            output[i] = new Instances(i_Data, 0, 0);
        }
        for (int i = 0; i < i_Data.numInstances(); i++) {
            Instance toAdd = i_Data.instance(i);
            int indexToAdd = (int) toAdd.value(i_AttributeIndex);
            output[indexToAdd].add(toAdd);
        }
        return output;
    }

    private boolean proneOrNot(Instances i_Data, int i_AttributeIndex)
    {
        boolean output;
        if(this.pValue == 1)
        {
            output = false;
        }
        else {
            double chiSquareValue = calcChiSquare(i_Data, i_AttributeIndex);
            int degreeOfFreedom = degreeOfFreedom(sortInstancesByAttributeValue(i_Data, i_AttributeIndex)) - 1;
            int alphaRiskIndex = -1;
            for(int i = 0; i < pValues.length; i++)
            {
                if(pValue == pValues[i])
                {
                    alphaRiskIndex = i - 1;
                }
            }
            double alphaRisk = chiTable[alphaRiskIndex][degreeOfFreedom - 1];
            output = chiSquareValue < alphaRisk;
        }
        return output;
    }

    /**
     * calculates the height in the tree of an instance classification
     * @param i_Instance
     */
    private int Height(Instance i_Instance)
    {
        int path = 0;
        Node currentNode = this.rootNode;
        while(currentNode.attributeIndex != -1){
            int currentAttribute = currentNode.attributeIndex;
            currentNode = currentNode.children[(int)i_Instance.value(currentAttribute)];
            if(currentNode == null) {
                return path;
            }
            path++;
        }
        return path;
    }
    //Visualization of the tree. Method made by Nitai
    public void printTreeTest(Instances data) {
        printTreeTest(data, "", rootNode);
    }
    private void printTreeTest(Instances data, String tab, Node node) {
        if (node.children != null) {
            if (node.parent == null) {
                System.out.println("Root - " + data.attribute(node.attributeIndex).name() + " - ("
                        + distributeInstances(data, node).numInstances() + ")");
                tab = tab + "|   ";
            }
            for (int i = 0; i < node.children.length; i++) {
                if (node.children[i] != null) {
                    System.out.println(tab + data.attribute(node.children[i].parent.attributeIndex).name() + " = "
                            + data.attribute(node.children[i].parent.attributeIndex).value((int)node.children[i].returnValue)
                            + ": " + data.attribute(9).value((int) node.children[i].returnValue) + " ("
                            + distributeInstances(data, node.children[i]).numInstances() + ")");
                    printTreeTest(data, tab + "|   ", node.children[i]);

                }
            }
        }
    }
    private Instances distributeInstances(Instances data, Node node) {
        Instances distributed = data;
        Node curr = node;
        LinkedList<Node> nodesList = new LinkedList<Node>();
        while (curr.parent != null) {
            nodesList.addFirst(curr);
            curr = curr.parent;
        }
        while (!nodesList.isEmpty()) {
            distributed = getChild(distributed, (int)nodesList.getFirst().returnValue, curr.attributeIndex);
            curr = nodesList.pollFirst();
            if (distributed.numInstances() == 0) {
                break;
            }
        }
        return distributed;
    }

    // set of instances with specific attribute
    private Instances getChild(Instances data, int val, int attIndex) {
        Instances child = new Instances(data, 0);
        for (int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).value(attIndex) == val || val == -1) {
                child.add(data.instance(i));
            }
        }
        return child;
    }
}
