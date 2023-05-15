
X=""
X1=123

#include_x1=true

if [ -z ${include_x1+x} ]; then echo "no x1"; else X+=" $X1"; fi

for x in $X
do
python test.py \
  --x "$x" \
  --y True
done
