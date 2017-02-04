declare -a content=("content0" "content1" "content2")
declare -a style=("style0" "style1" "style2" "style3" "style4" "style5")

for i in "${content[@]}"
do
  for j in "${style[@]}"
   do
   python3 transfer.py -style_image_path images/$j.jpg -content_image_path images/$i.jpg -style_name $j -content_name $i
   done
done
