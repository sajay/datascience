Using the same docker image as before. Ignore steps 1 & 2 if you have the image build already.

Set up your environment with the commonly used DS packages

1)docker pull kaggle/python

https://github.com/Kaggle/docker-python

This is around 25 GB

More info here : http://blog.kaggle.com/2016/02/05/how-to-get-started-with-data-science-in-containers/

2)docker image ls

3)CD to the datascience/algorithms folder & run the container with the name 'ds-cname'

docker run -v $PWD:/tmp/working -w=/tmp/working --rm -it -d --name ds-cname kaggle/python:latest

4)docker container ls

5)Docker container details (If needed)

-docker container inspect ds-cname
Get the IP address of the container ds-cname: docker exec ds-cname cat /etc/hosts

6)For the Naive bayes algorithm:

a)This is from lesson2 of this course: https://classroom.udacity.com/courses/ud120

b)Run the classifier from the container : 
  docker exec -it ds-cname python3 naivebayes/nb_author_id.py

7)After finish :

docker container stop ds-cname

