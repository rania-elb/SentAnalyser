
#my $fileName;
my $file;
my $cleanedFile;

my $line1;
my $line2;
my $prefixe1;
my $prefixe2;
my $suffixe1;
my $suffixe2;

open($file, "<train2.tsv");
open($cleanedFile, ">train_cl.tsv");

################
# Header

$line1 = <$file>;
if($line1 =~ s/^\t//){}
print $cleanedFile $line1;

################
# Cleaning

#first $line1 iteration

$line1 = <$file>;

if($line1 =~ /^[0-9]+\t([0-9]+\t[0-9]+\t)(.*)(\t[0-4]\n$)/){
	$prefixe1 = $1;
	$line1 = " ".$2." ";
	$suffixe1 = $3;
}
	# spaces
if($line1 =~ s/ +/ /g){}
if($line1 =~ s/^ | $//g){}

while($line2 = <$file>){
	if($line2 =~ /^[0-9]+\t([0-9]+\t[0-9]+\t)(.*)(\t[0-4]\n$)/){
		$prefixe2 = $1;
		$line2 = " ".$2." ";
		$suffixe2 = $3;
	}
	
	# spaces
	if($line2 =~ s/ +/ /g){}
	if($line2 =~ s/^ | $//g){}
	
	if($line2 ne ""){
		if($line1 ne $line2){
			print $cleanedFile $prefixe1.$line1.$suffixe1;
		}
		
		$prefixe1 = $prefixe2;
		$line1 = $line2;
		$suffixe1 = $suffixe2;
	}
}

close($file);
close($cleanedFile);

