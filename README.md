# PerceptronIA
Questo progetto è stato sviluppato per l'esame di Intelligenza Artificiale del corso di Laurea in Ingegneria Informatica presso l'università di Firenze tenuto dal Prof. Paolo Frasconi.

In questo progetto sono stati implementati gli algoritmi Perceptron e VotedPerceptron con l'obiettivo di confrontarne le prestazioni analizzando la matrice di confusione e l'accuratezza su Dataset con almeno 1000 esempi ciascuno.

## UCI Machine Learning Repositories
Sono stati reperiti 3 Dataset dall'UCI Machine Learning Repository:
* Diabetic Retinopathy Debrecen Dataset https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set 
* HTRU2 Dataset https://archive.ics.uci.edu/ml/datasets/HTRU2 
* Banknote Authentication Dataset https://archive.ics.uci.edu/ml/datasets/banknote+authentication

## Program Structure
Il programma è strutturato in 3 elementi principali:
* I files contententi l'implementazione degli algoritmi Perceptron e VotedPerceptron;
* I file contententi l'implementazione dei Dataset:
  * Dataset per importare i dati,
  * TestingDataset per eseguire i test,
  * PlotDataset per vedere i risultati;
* Il file TestingAll

## Development 
Come ambiente di sviluppo è stato utilizzato PyCharm e il linguaggio di programmazione usato è Python

## Utilizzo
Per ottenere la matrice di confusione e l' accuratezza inerenti ai risultati ottenuti confrontando il Perceptron e il VotedPerceptron sui dataset sopracitati è sufficiente eseguire TestingAll.py

In questo sono organizzati i metodi di plotting dei Dataset che chiamano a loro volta il procedimento di training e testing degli algoritmi sui dataset elaborati nei files corrispondenti.

## Approfondimenti e Risultati
Per ulteriori specifiche inerenti all'implementazione rimandiamo al file PDF che illustra in dettaglio il funzionamento del progetto e in cui sono anche riportate considerazioni sui risultati.
