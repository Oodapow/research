# What is this?

In this folder you can find the tools needed for a smooth developing experience on your ML projects.

## Motivation

For small projects a google colab notebook might be enough to handle the workload. Maybe a laptop with a GPU might be enough. But when the training time is days or more and the number of experiments starts growing, better tools are needed.

## Structure

The tools are divided into 3 Layers:
 * machine layer
 * cluster layer
 * application layer

### Machine Layer

The machine layer contains any infrastructure component that requires you to `ssh` into a specific machine for the setup. In contrast, the cluster layer and application layer interact with the machines with the kubernetes API, this can be done directly form a well configured laptop.


Here you can find scripts that install the following tools:
 * python - some of the cluster/config layer scripts are in python to provide an OS agnostic experience.
 * NFS - a network file server is needed to persist data across multiple computers. This avoids the need of a manual sync between the systems involved in a project.
 * postgresql - a database that will be needed by some applications. Even if most applications have a way of spinning a sqldb it is better to use a dedicate one that you manage yourself. This allows the persistence of the database to be independent of the cluster persistence.
 * kubernetes - here are the scripts that install the tools needed to set up a cluster, but also scripts that help out with the setting up process. Here you can also find the way to configure your laptop to access the cubernetes cluster.

### Cluster Layer

The cluster layer contains any infrastructure component that is required for easy interaction with the kubernetes cluster.


Here you can find scripts and readmes that (show how to) install the following tools:
 * helm - this is installed on the administrative laptop and will be used by the application layer to install other tools.
 * dashboard - this is the easy way of interacting with the kubernetes cluster. It offers most of the functionality of the `kubectl` tool, but in an UI manner. All UI interactions have a `kubectl` equivalent, sometimes the dashboard even tells you the line.
 * volumes - by default a kubernetes cluster has no way of dealing with persistent data. This can be solved by adding a `StorageClass` API object in the cluster and setting it as default, the alternative being the manual administration of `PersistentVolume` API objects. To solve this issue a NFS provisioner is installed into the cluster.
 * load balancing - by default the cluster has no support for load balancing. This limits the service type to `NodePort` meaning a service can be exposed only by oppening a port on one of the nodes. MetalLB is one of the solutions that can be used to obtain this functionality.

### Application Layer

The application layer contains any infrusturcture component that can be installed with `helm`.


A list of good tools to have in a private kubernetes cluster for a ML project is:
 * minio - data at a higher level that a NFS server.
 * gpu monitoring - see the usage of yor GPUs
 * git - version you code. This would be used in addition the other git tools for containing the code of all the experiments (including the failed ones), while the main git repo would contain the code of the success models.
 * polyaxon - ML at scale, offers a way to reproduce experiments and do hyperparameter searches. It also offers a simple way of interacting with the kubernetes cluster for starting machine learning servers like jupyter and tensorboard.
 * spark - a tool that focuses on big tensors that would not be able to fit in the memory of a single machine.
