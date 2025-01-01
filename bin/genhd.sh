
echo "#pragma once"
echo "#define _DEFAULT_KERNEL_DIR_PATH_ \"$1\""

get_md5=`git rev-parse HEAD`

num_mod_files=$(git diff --name-only | wc -l)
if [ "$num_mod_files" -gt 0 ]; then
    num_mod_files=1
else
    num_mod_files=0
fi

echo "#define _PIMBLAS_GIT_VERSION_ \"$num_mod_files::$get_md5\""
echo "#define _PIMBLAS_VERSION_ \"0.1\""






