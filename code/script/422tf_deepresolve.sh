dir="$1"
ftmap="$2"
scmap="$3"
nm="$4"
ds="$5"
suf="$6"
#wt="$7"

python 1_allmotifgen_DR_weighted_oniv.py ${dir} ${dir} ${ftmap} ${scmap} 16_G_ ${nm} ${ds}
# ${wt}
python 2_match_DR.py ${dir} ${dir} ${nm}
python 3_evalu_ext.py ${dir} ${dir} match_${nm} ${suf}
python 4_count_percent_all.py E-value${suf} count${suf} 

