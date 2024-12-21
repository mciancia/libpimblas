
echo "#pragma once"
echo "#define _DEFAULT_KERNEL_DIR_PATH_ \"$1\""

get_md5=`git rev-parse HEAD`

num_mod_files=`git diff --name-only | wc -l`

echo "#define _PIMBLAS_GIT_VERSION_ \"$num_mod_files::$get_md5\""







