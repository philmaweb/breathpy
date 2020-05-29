Origin and sources: http://genomeinformatics.uni-due.de/research/ion-mobility-spectroscopy-ims-analysis-with-restricted-resources/

Author: Marianna D'Addario, Dominik Kopczynski
E-Mail: {Marianna.Daddario, Dominik.Kopczynski}@tu-dortmund.de
Copyright (c) 2013 Marianna D'Addario, Dominik Kopczynski

Please note that the here included yoshiko is a software from http://www.cwi.nl/research/planet-lisa and consider the license agreement made for yoshiko. 

USAGE:
./peax measurement_ims.csv result.csv
eg
./peax BD18_1408280834_ims.csv BD18_1408280834_ims_out.csv
- measurement_ims.csv is an MCC/IMS measurement. Please find some anonymized measurements on our website https://sites.google.com/site/rahmannlab/research/ims .

- result.csv is tab-separated and reports every found peak within measurement_ims.csv.

Please check the content of parameters.cfg to create the desired pipeline instance and to adjust the given parameters.
