Spark Workshop Demo Instructions
=================================

These are the instructions for following the Spark workshop presented by Hortonworks. The recording of workshop can be found 
[here](https://www.brighttalk.com/webcast/9573/140773). By following the instructions on this page, you will be able to:

1. Install Spark and execute Spark jobs on the Hortonworks Data Platform (HDP)
2. Understand how to use Spark's main elements -- Core API, Spark SQL, Spark MLlib and Spark Streaming 
3. Understand how Spark can be used with Hive 

Prerequisites
-------------

0. Download the HDP 2.2 Sandbox from [here](http://hortonworks.com/products/hortonworks-sandbox/#install)
1. Download and install Spark 1.2 technical preview on the sandbox by following the instructions on this [page](http://hortonworks.com/hadoop-tutorial/using-apache-spark-hdp/)
2. Log into the sandbox, and clone this repository.  

```bash
$ su hdfs
$ cd
$ git clone https://github.com/DhruvKumar/spark-workshop
```

3. Download the sample tweet data into the sandbox

```bash
$ wget 
```

4. Put the tweet data into hdfs, at a location /tmp/tweets

```bash
$ hadoop fs -put tweets /tmp/tweets
```

Explore the data using Spark SQL and Hive
-----------------------------------------

We are now ready to start data analysis. Go to Spark install directory's bin folder, and launch the Spark shell:

```bash
$ ./spark-shell --master yarn-client --driver-memory 512m --executor-memory 512m
```

This will launch the Spark shell and give you a REPL from where you can execute Spark commands:

```bash
15/03/19 00:09:34 INFO AkkaUtils: Connecting to HeartbeatReceiver: akka.tcp://sparkDriver@sandbox.hortonworks.com:45177/user/HeartbeatReceiver
15/03/19 00:09:34 INFO NettyBlockTransferService: Server created on 58740
15/03/19 00:09:34 INFO BlockManagerMaster: Trying to register BlockManager
15/03/19 00:09:34 INFO BlockManagerMasterActor: Registering block manager localhost:58740 with 265.4 MB RAM, BlockManagerId(<driver>, localhost, 58740)
15/03/19 00:09:34 INFO BlockManagerMaster: Registered BlockManager
15/03/19 00:09:34 INFO SparkILoop: Created spark context..
Spark context available as sc.

scala>
```

Next, create the HiveContext and SQLContext objects. Spark Context is already created by the spark shell and is available to use in a value "sc." We can use Spark Context object to create the Hive Context and Spark SQL Context:

```scala
scala> val hiveCtx = new org.apache.spark.sql.hive.HiveContext(sc)

scala> val sqlCtx = new org.apache.spark.sql.SQLContext(sc)
```

Now, let's create a RDD from the tweet file:

```scala
scala> val tweetTableHive = hiveCtx.jsonFile("/tmp/tweets")
```

You can examine its schema and cache it:

```scala
scala> tweetTableHive.printSchema()

scala> tweetsTableHive.cache()
```
Why did we cache the RDD? We will be running Hive queries against this dataset soon so it is useful to store it in memory for faster query processing. 

Now, let's register it as a temporary table and assign it a name:

```scala
scala> tweetTableHive.registerTempTable("tweetTableHive")
```
Let's query some data and make sure that the table is set up correctly. One simple query is to print 100 tweet texts:

```scala
hiveCtx.hql("select text from tweetTableHive limit 100").collect().foreach(println)
```
You should see some tweets printed on the screen. Next, let's try to store this data into a Hive Table. Spark's built-in SQL manipulation objects such as HiveContext and SparkSQLContext have a seamless integration with Hive. For storing data in Hive, we will use the Optimized Row Columnar (ORC) format instead of text. ORC is a compressed file format which gives Hive a great performance boost (100x). The next few commands whill show you how you can create a Hive tabel from Spark, store data in it and query it.  

```scala
scala> hiveCtx.hql("create table orc_tweets(user string, text string) stored as orc")

scala> hiveCtx.hql("insert into table orc_tweets select user.name, text from tweetTableHive")

scala> hiveCtx.hql("select user, text from orc_tweets limit 1").collect().foreach(println)
```

You should be able to see tweets displayed in the Scala REPL. So far we've parsed data from JSON, stored it in Hive and queried it. Let's do some Machine Learning on it.

Using Spark MLlib on Spark SQL
------------------------------

To prepare our dataset in Spark SQL, let's bring the table back in memory. These steps are identical to what we did with Hive Context.

```scala
scala> val tweetTableSql = sqlCtx.jsonFile("/tmp/tweets").cache()

scala> tweetTableSql.registerTempTable("tweetTableSql")

scala> val texts = sqlCtx.sql("select text from tweetTableSql").map(_.head.toString)
```

At this point, we've created a SchemaRDD called `texts`. This RDD contains the text of each tweet. Our next step is to convert each text contained in this RDD to a numerical representation which can be used as an input to our machine learning algorithm. 

```scala
scala> import org.apache.spark.mllib.feature.HashingTF

scala> import org.apache.spark.mllib.clustering.KMeans

scala> val tf = new HashingTF(1000)

scala> def featurize(s: String): Vector = { tf.transform(s.sliding(2).toSeq) }
```

The `featurize(s: String)` function can take any string and hash two characters at a time, forming a bigram model of our text. We've essentially used the [Hashing Trick](http://en.wikipedia.org/wiki/Feature_hashing) here. Now, let's convert our tweets into feature vectors using the `featurize` function:

```scala

scala> val vectors = texts.map(featurize).cache() 

```
The machine learning algorithm we're going to use is called [k-means](http://en.wikipedia.org/wiki/K-means_clustering). K-Means is an unsupervised algorithm--it is used for quick exploration of data sets before applying more complex algorithms. Often, it is used by Data Scientists for dividing the data set into separate classes and to "get a feel for the data." For our purposes, this is a good algorithm since we're interested in dividing the text of tweets into some clusters, say 10.

Let's run the k-means algorithm present in Spark MLlib. We're telling the KMeans algorithm to divide our `vectors` into 10 clusters, and run a maximum of 20 times to converge.

```scala

scala> val model = KMeans.train(vectors, 10, 20)

```

This should take a few seconds. Once the training finishes, you can see what type of clusters the model has produced. 



```scala

scala> val some_tweets = texts.take(100)

scala> for (i <- 0 until 10) {
        println(s"\nCLUSTER $i:")
        some_tweets.foreach { t =>
          if (model.predict(featurize(t)) == i) {
            println(t)
        }
      }
    }

```


```scala

scala> sc.makeRDD(model.clusterCenters, 10).saveAsObjectFile("/tmp/modelSmall")

```
