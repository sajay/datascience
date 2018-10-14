Set up your environment with the commonly used DS packages

1)docker pull kaggle/python

https://github.com/Kaggle/docker-python

This is around 25 GB

More info here : http://blog.kaggle.com/2016/02/05/how-to-get-started-with-data-science-in-containers/


2)docker image ls

3)CD to the datascience/titanic_analysis folder & run the container with the name 'ds-cname'

docker run -v $PWD:/tmp/working -w=/tmp/working --rm -it -d --name ds-cname kaggle/python:latest

4)docker container ls

5)docker container inspect ds-cname

6)Get the IP address of the container ds-cname:
  
docker exec ds-cname cat /etc/hosts

(See the last line)
 
7)Open the container to confirm :

docker exec -it ds-cname sh

8)Execute the python src file:

docker exec -it ds-cname python3 src/process_input.py

9)After finish :

docker container stop ds-cname



