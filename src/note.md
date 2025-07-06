# Note del progetto : 
- **Filter approaches**: Features are selected before the data mining algorithm is run. The specific filtering method might be independent of the learning algorithm. For instance, we may remove features with little correlation with the target, or remove a feature that is highly correlated with another one (it is redundant).

 - **Wrapper approaches**: Feature are selected on the basis of their contribution, and the contribution is measured after running the algorithm as a black-box.



# azioni da fare : 

1) split train & test 

2) eliminare le colonne del train cui c'è una quantità di NaN maggiore del 50% 

3) fare correlazione fra le colonne ed eliminare tutte le colonne con una forte correlazione fra di loro e una bassa correlazione con il target. (FILTER APPROACH)
4) Usare un wrapper Approach per una semiconferma (va bene anche un modello lineare multivariato).

