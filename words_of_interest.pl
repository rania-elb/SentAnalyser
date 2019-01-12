
#my $fileName;
my $file;
my $wordsFile;
my $wordsFile2;
my $vocaFile;

my $line1;
my $line2;
my $sent1;
my $sent2;
my $id1;
my $id2;
my $word;

#open($vocaFile, "<voca3.txt");
open($file, "<train.tsv");
open($wordsFile, ">wordsSents.csv");
open($wordsFile2, ">wordsNeutrals.csv");

################
# Header

$line1 = <$file>;

################
# Cleaning

#first $line1 iteration

$line1 = <$file>;

if($line1 =~ /^([0-9]+)\t[0-9]+\t(.*)\t([0-4])\n$/){
	$id1 = $1;
	$line1 = " ".$2." ";
	$sent1 = $3;
}

my $i = 0;

print $wordsFile "PhraseId\tWord\tSentiment\n";
print $wordsFile2 "PhraseId\tWord\n";

while($line2 = <$file>){
	if($line2 =~ /^([0-9]+)\t[0-9]+\t(.*)\t([0-4])\n$/){
		$id2 = $1;
		$line2 = " ".$2." ";
		$sent2 = $3;
	}
	
	if($sent2 eq 2 and $line1 =~ /($line2)([^ ]+)/){
		$i += 1;
		if($sent1 ne 2){
			$word = $2;
			print $wordsFile "$id1\t$word\t$sent1\n";
		}elsif($sent1 eq 2){
			$word = $2;
			print $wordsFile2 "$id1\t$word\n";
		}
	}
	
	$id1 = $id2;
	$line1 = $line2;
	$sent1 = $sent2;
}

print "\n$i lignes traitées.\n";

#close($vocaFile);
close($file);
close($wordsFile);
close($wordsFile2);

