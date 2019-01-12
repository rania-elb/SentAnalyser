cut -f2 wordsNeutrals.csv | sort -f | uniq -i -c | sed -e 's/^ *//' | sed -e 's/ /\t/' > vocabNeutral.csv
cut -f2 wordsSents.csv | sort -f | uniq -i -c | sed -e 's/^ *//' | sed -e 's/ /\t/' > vocabSents.csv
