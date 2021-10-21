# Check and install packages if necessary
for (package in c('devtools', 'readr', 'optparse')) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package)
  }
}

library(readr)
library(optparse)
library(devtools)

if (!require(robROSE)) {
  devtools::install_github("SebastiaanHoppner/robROSE/robROSE")
}

library(robROSE)

# Parser for command line arguments
option_list = list(
  make_option(c("-f", "--file"), type="character", default=NULL, 
              help="dataset file name", metavar="character"),
  make_option(c("-o", "--out"), type="character", default=NULL, 
              help="output file name", metavar="character"),
  make_option(c("-l", "--label"), type="character", default=NULL,
              help="Label column for dataset", metavar="character"),
  make_option(c("-r", "--r"), type="double", default=0.2,
              help="Desired fraction of minority cases [default= %default", metavar="double"),
  make_option(c("-a", "--alpha"), type="double", default=0.5,
              help="Numeric parameter used by the covMcd function for controlling the size of the subsets over which the determinant is minimized [default= %default]", metavar="double"),
  make_option(c("-c", "--const"), type="double", default=1,
              help="Tuning constant that changes the volume of the elipsoids [default= %default]", metavar="double"),
  make_option(c("-s", "--seed"), type="integer",
              help="A single value, interpreted as an integer, recommended to specify seeds and keep trace of the generated sample.", metavar="integer")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$file)) {
  print_help(opt_parser)
  stop('Dataset file name must be supplied')
} else if (is.null(opt$out)) {
  print_help(opt_parser)
  stop('Output file name must be supplied')  
} else if (is.null(opt$label)) {
  print_help(opt_parser)
  stop('Label column must be supplied')  
} 


# Reads in data and samples based on given arguments
df = read_csv(opt$file)
f_str = paste(c(opt$label, '.'), collapse=' ~ ')
f = as.formula(f_str)
df_rob = robROSE(f, data=df, r=opt$r, alpha=opt$alpha, const=opt$const, seed=opt$seed)
write.csv(df_rob$data, file=opt$out)

q('no')