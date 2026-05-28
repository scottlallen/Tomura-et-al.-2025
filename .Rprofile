source("renv/activate.R")
# 1. Force renv to use base R's download manager
Sys.setenv(RENV_DOWNLOAD_METHOD = "utils")

# 2. Force R to use Windows' standalone curl.exe instead of internal libcurl
options(download.file.method = "curl")
options(download.file.extra = "--insecure")

# 3. Use the standard secure CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))