#!/usr/bin/env bash
cd "$(dirname "$0")"

# Δημιουργία φακέλων
mkdir -p temp output/NAC output/AC

# Templates
TEMPLATE_NAC="PET_NAC_single.mac"
TEMPLATE_AC="PET_AC_single.mac"

# Έλεγχος αν υπάρχουν τα templates
if [[ ! -f $TEMPLATE_NAC ]] || [[ ! -f $TEMPLATE_AC ]]; then
  echo "Λείπουν τα template αρχεία PET_NAC.mac ή PET_AC.mac!"
  exit 1
fi

# Μετρητής runs
count=92

# Loop σε κάθε γραμμή του parameters.txt
while IFS=' ' read -r R_PHANTOM R_SOURCE X_POS Y_POS Z_POS cm; do
    echo "Run $count → R=$R_PHANTOM, SR=$R_SOURCE, X=$X_POS, Y=$Y_POS, Z=$Z_POS"

    # NAC αρχείο (μέσα στον φάκελο temp)
    sed \
      -e "s#__OUTPUT_NAME__#output/NAC/img_${count}#g" \
      -e "s#__R__#${R_PHANTOM}#g" \
      -e "s#__SR__#${R_SOURCE}#g" \
      -e "s#__X__#${X_POS}#g" \
      -e "s#__Y__#${Y_POS}#g" \
      -e "s#__Z__#${Z_POS}#g" \
      -e "s#__cm__#${cm}#g"\
      $TEMPLATE_NAC > "temp/run_${count}_nac.mac"
    Gate temp/run_${count}_nac.mac

    
    # AC αρχείο (μέσα στον φάκελο temp)
    sed \
      -e "s#__OUTPUT_NAME__#output/AC/img_${count}#g" \
      -e "s#__R__#${R_PHANTOM}#g" \
      -e "s#__SR__#${R_SOURCE}#g" \
      -e "s#__X__#${X_POS}#g" \
      -e "s#__Y__#${Y_POS}#g" \
      -e "s#__Z__#${Z_POS}#g" \
      -e "s#__cm__#${cm}#g"\
      $TEMPLATE_AC > "temp/run_${count}_ac.mac"
    Gate temp/run_${count}_ac.mac

    echo "Created temp/run_${count}_nac.mac and temp/run_${count}_ac.mac"

    # Αύξηση μετρητή
    count=$((count + 1))
done < parameters_single.txt

echo "✅ Όλα τα MAC αρχεία δημιουργήθηκαν στον φάκελο temp!"







