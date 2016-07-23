package com.learning.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator$;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tree.impl.RandomForest;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF3;
import org.apache.spark.sql.api.java.UDF4;
import org.apache.spark.sql.catalyst.expressions.IsNotNull;
import org.apache.spark.sql.types.DataTypes;

import java.util.Map;

import static org.apache.spark.sql.functions.*;

/**
 * Created by sauraraj on 6/23/2016.
 */
public class TitanicProblemSpark {

    static SparkConf sparkConf = new SparkConf()
            .setMaster("local[*]")
            .setAppName("Titanic Spark");
    static JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);
    static SQLContext sqlContext = new SQLContext(javaSparkContext);

    public static void main(String[] args) {
        // Spark on windows looks for winutils.exe file in Hadoop Home. So we need to set Hadoop home dir.
        System.setProperty("hadoop.home.dir", "C:\\hadoop-2.6.4");
        String trainFile = "D:\\Documents\\kaggleQuestions\\titanicProblem\\dataset\\train.csv";
        String testFile  = "D:\\Documents\\kaggleQuestions\\titanicProblem\\dataset\\test.csv";
        //String resultsFile = "D:\\Documents\\kaggleQuestions\\titanicProblem\\results\\result_Spark_RFC.csv";

        DataFrame inputTrainDF = sqlContext.read().format("com.databricks.spark.csv")
                .option("header","true").option("inferSchema","true").load(trainFile);

        System.out.println("Train file count: " + inputTrainDF.count());

        DataFrame inputTestDF = sqlContext.read().format("com.databricks.spark.csv")
                .option("header","true").option("inferSchema","true").load(testFile);
        // Pre-processing of data which involves filling na values, dropping unrequired columns, creating new features etc.
        DataFrame trainDF = processData(inputTrainDF);
        trainDF.show();
        // Converting string labels into indices.
        StringIndexer embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndexed").setHandleInvalid("skip");
        StringIndexer sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndexed").setHandleInvalid("skip");
        StringIndexer survivedIndexer = new StringIndexer().setInputCol("Survived").setOutputCol("SurvivedLabel").setHandleInvalid("skip");
        // Creating dummy columns using One hot encoder.
        OneHotEncoder embEncoder = new OneHotEncoder().setInputCol("EmbarkedIndexed").setOutputCol("EmbarkedVec");
        OneHotEncoder sexEncoder = new OneHotEncoder().setInputCol("SexIndexed").setOutputCol("SexVec");
        // The vector assembler creates a feature column where it combines all the required features at one place.
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"Pclass","SexVec","Age","SibSp","Parch","Fare","EmbarkedVec","Family","Child","Mother"})
                .setOutputCol("features");
        // Performing PCA
        PCA pca = new PCA().setK(7).setInputCol("features").setOutputCol("pcaFeatures");

        RandomForestClassifier rfc = new RandomForestClassifier()
                .setFeaturesCol("pcaFeatures")
                .setLabelCol("SurvivedLabel");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{embarkedIndexer, sexIndexer, survivedIndexer, embEncoder, sexEncoder, assembler, pca, rfc});
        // Specifying grid search parameters
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(rfc.maxBins(),new int[]{25,50,100})
                .addGrid(rfc.maxDepth(),new int[]{4,6,10})
                .build();

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("SurvivedLabel");
        // Cross Validation using K-folds approach
        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5);

        CrossValidatorModel crossValidatorModel = crossValidator.fit(trainDF);
        DataFrame bestModel = crossValidatorModel.bestModel().transform(trainDF);
        bestModel.select("PassengerId","prediction").show(30);
        System.out.println("Before best model");
        // Processing Test data
        DataFrame testDF = processData(inputTestDF);
        // Prediction on test data.
        DataFrame predTest = crossValidatorModel.transform(testDF);
        predTest = predTest.withColumn("prediction", predTest.col("prediction").cast("int"));
        predTest = predTest.withColumnRenamed("prediction","Survived");
        predTest.select("PassengerId","Survived").show(30);
        // Saving to a File
        predTest.coalesce(1)
                .select("PassengerId", "Survived")
                .write().mode("Overwrite")
                .format("com.databricks.spark.csv")
                .option("header","true")
                .save("res.csv");



    }

    public static DataFrame processData(DataFrame data){
        data = data.drop("Ticket").drop("Cabin");
        Double ageMean = data.select(mean("Age")).head().getDouble(0);
        Double fareMean = data.select(mean("Fare")).head().getDouble(0);
        data = data.na().fill(ageMean, new String[]{"Age"});
        data = data.na().fill(fareMean, new String[]{"Fare"});
        data = data.withColumn("Family",data.col("Parch").$plus(data.col("SibSp")));

        sqlContext.udf().register("childCol", new UDF1<Double, Integer>() {
            @Override
            public Integer call(Double val){
                if (val<18)
                    return 1;
                else
                    return 0;
            }
        }, DataTypes.IntegerType);


        sqlContext.udf().register("motherCol", new UDF4<Double, String, Integer, String, Integer>() {
            @Override
            public Integer call(Double age, String gender, Integer parch, String name ){
                if ((age > 18)&&(gender.equals("female"))&&(parch > 0)&&(!name.contains("Miss")))
                    return 1;
                else
                    return 0;
            }
        },DataTypes.IntegerType);

        data = data.withColumn("Child",callUDF("childCol",data.col("Age")));
        data = data.withColumn("Mother", callUDF("motherCol",data.col("Age"),data.col("Sex"),data.col("Parch"),data.col("Name")));
        //data = data.drop("Name");

        return data;
    }
}
