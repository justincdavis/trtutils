.. _installation:

Installation
------------

There are multiple methods for installing trtutils. The recommended method is
to install trtutils into a virtual environment. This will ensure that the
trtutils dependencies are isolated from other Python projects you may be
working on.

Methods
^^^^^^^
#. Pip:

   .. code-block:: console

      $ pip3 install trtutils

#. From source:

   .. code-block:: console

      $ git clone https://github.com/justincdavis/trtutils.git
      $ cd trtutils
      $ pip3 install .

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

#. dev:

   .. code-block:: console

      $ pip3 install trtutils[dev]
   
   This will install dependencies allowing a full development environment.
   All CI and tools used for development will be installed and can be run.
