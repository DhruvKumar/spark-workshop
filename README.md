Spark Workshop Demo Instructions
=================================

These are the instructions for following the Spark workshop which I presented on March 12, 2015. The recording of workshop can be found 
[here](https://www.brighttalk.com/webcast/9573/140773). By following the instructions on this page, you will be able to:

1. Install Spark and execute Spark jobs on the Hortonworks Data Platform (HDP)
2. Understand how to use Spark's main elements -- Core API, Spark SQL, Spark MLlib and Spark Streaming 
3. Understand how Spark can be used with Hive 

Prerequisites
-------------

0. Download the HDP 2.2 Sandbox from [here](http://hortonworks.com/products/hortonworks-sandbox/#install)
1. Download and install Spark 1.2 technical preview on the sandbox by following the instructions on [this page](http://hortonworks.com/hadoop-tutorial/using-apache-spark-hdp/)
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

5. Sign up for dev Twitter account and get the OAuth credentials [here for free](https://apps.twitter.com/). You need these credentials to use the Spark Streaming poriton of this workshop, which takes place at the end. You can skip this step if you're only interested in Spark SQL, Hive Context and MLlib portions of this workshop. 

Explore the data using Spark SQL and Hive
-----------------------------------------

We are now ready to start data analysis. Go to Spark install directory's bin folder, and launch the Spark shell:

```bash
$ ./spark-shell --master yarn-client --driver-memory 512m --executor-memory 512m
```

This will launch the Spark shell and give you a read-eval-print-loop (REPL) from where you can execute Spark commands:

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

Next, create the HiveContext and SQLContext objects. Spark Context is already created by the spark shell and is available to use in the object `sc`. We can use Spark Context object to create the Hive Context and Spark SQL Context:

```scala
scala> val hiveCtx = new org.apache.spark.sql.hive.HiveContext(sc)

scala> val sqlCtx = new org.apache.spark.sql.SQLContext(sc)
```

Now, let's create a RDD from the tweet file:

```scala
scala> val tweetTableHive = hiveCtx.jsonFile("/tmp/tweets")
```

You can examine RDD's schema and cache it:

```scala
scala> tweetTableHive.printSchema()

scala> tweetsTableHive.cache()
```
Why did we cache the RDD? We will be running Hive queries against the RDD soon so it helps to store it in memory for faster query processing. 

Now, let's register the RDD as a temporary table and assign it a name:

```scala
scala> tweetTableHive.registerTempTable("tweetTableHive")
```
Let's query some data and make sure that the table is set up correctly. One simple query is to print 100 tweets:

```scala
hiveCtx.hql("select text from tweetTableHive limit 100").collect().foreach(println)
```
You should see some tweets printed on the screen. Next, let's try to store this data into a Hive Table. Spark's built-in SQL manipulation objects such as HiveContext and SparkSQLContext have a seamless integration with Hive. For storing data in Hive, we will use the Optimized Row Columnar (ORC) format instead of text. ORC is a compressed file format which gives Hive a great performance boost (100x). The next few commands whill show you how you can create a Hive table from Spark, store data in it and query it.  

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

Note the cluster numbers and the language associated with each cluster. You'll find that similar looking tweet languages are clustered in groups. Pick the cluster number associated with your favorite language. One good choice is Arabic since k-means is able to group Arabic strings in a distinct cluster very well. Note down the number of the cluster, we will use it when we apply the model to a live twitter stream.

Store the model in HDFS at a location `/tmp/modelSmall`

```scala
scala> sc.makeRDD(model.clusterCenters, 10).saveAsObjectFile("/tmp/myModel")
```

Now let's apply the model to a live twitter stream.

Using Spark Streaming 
---------------------

We'll run the spark streaming code against a live twitter stream. Our objective is to filter out and print only those tweets whose language matches our favorite language.

Before running the code, let's compile it first. Go to the root directory of this workshop's code on your HDP Sandbox, and type:

```bash
$ cd twitter_classifier/scala/
$./sbt/sbt compile
```

Assuming `{SPARK_HOME}` is the directory where you un-tarred the Spark tarball on your Sandbox, you can run the compiled Spark streaming code as follows:

```bash
$ cd 
$ {SPARK_HOME}/bin/spark-submit \
    --class "dhruv.sparkdemo.Predict" \
    --master yarn-cluster \
    --num-executors 2 \
    --executor-memory 512m \
    --executor-cores 2 \
    ./target/scala-2.10/spark-twitter-lang-classifier-assembly-1.0.jar  \
    /tmp/myModel \ 
    {your favorite language cluster} \
    --consumerKey {your Twitter consumer key} \
    --consumerSecret {your Twitter consumer secret} \
    --accessToken {your Twitter access token} \
    --accessTokenSecret {your Twitter access token secret}
```

This command will set up spark streaming, connect to twitter using your dev credentials, and start filtering out the tweets which belong to the language category you've specified using `{your favorite language cluster}` argument. The `OAuth` Twitter credentials need to be specified as well for the app to connect to Twitter. You should have set these up while finishing the Prerequisites section shown on the top of this page. 

To see the app in action, navigate to the Resource Manager UI at http://sandbox.hortonworks.com:8088. If you haven't edited your host machine's `/etc/hosts` file to resolve `sandbox.hortonworks.com` to the IP address of the Sandbox, you should just use the IP address of the sandbox to get to the resource manager UI. Eg: http://172.169.130.30:8088. Once in the RM UI, find the application named "dhruv.sparkdemo.Predict" in the app list. Here's what it looks like in my environment:

![alt text](https://github.com/DhruvKumar/spark-workshop/)


Click on the application ID URL, which will take you to the Application Master:

![alt text](https://github.com/DhruvKumar/spark-workshop)

Click on "logs" in the rightmost column, and you should see the code filtering out the language you specified. I used Arabic, and here's the screenshot from my environment:

![alt text](https://github.com/DhruvKumar/spark-workshop)

To kill the application, simply control-c'ing it won't work. You still need to shut down the executor process which is streaming the data. That's happening in a separate thread. To do this, you'll need to use Yarn's management command. First, identify the app id:

```bash
$ yarn application -list
```

Next kill the app:

```bash
$ yarn application -kill <appId>
```

This completes the workshop.
