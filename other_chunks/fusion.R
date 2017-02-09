setwd("~/Documents/datacamp/scrap")


df <- data.frame()
for (dir in list.dirs(getwd())){
  print(dir)
  if (dir !="/home/valentin/Documents/datacamp/scrap")
    for (file in list.files(dir)){
      print(file)
      don <- read.csv(paste(dir, "/",file, sep = "", collapse = ""))
      if (length(names(don))==17){
        don <- don[, c("screenName", 'text')]
        colnames(don)[1] <- "user__screen_name"
      }
      don <- don[, c("user__screen_name", 'text')]
      df <- rbind.data.frame(df, don)
    }
}

write.csv(df, file = "fusion_semaine3_R.csv", colnames = TRUE)
