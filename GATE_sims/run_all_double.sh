#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p temp output/NAC output/AC
TEMPLATE_NAC="PET_NAC_double.mac"
TEMPLATE_AC="PET_AC_double.mac"
count=1

while IFS=' ' read -r \
    R1 SR1 X1 Y1 Z1 CM1 \
    R2 SR2 X2 Y2 Z2 \
    CM2
do
 

  # NAC
  sed \
    -e "s#__OUTPUT_NAME__#output/NAC/img_${count}#g" \
    -e "s#__R1__#${R1}#g" \
    -e "s#__SR1__#${SR1}#g" \
    -e "s#__X1__#${X1}#g" \
    -e "s#__Y1__#${Y1}#g" \
    -e "s#__Z1__#${Z1}#g" \
    -e "s#__cm__#${CM1}#g" \
    -e "s#__R2__#${R2}#g" \
    -e "s#__SR2__#${SR2}#g" \
    -e "s#__X2__#${X2}#g" \
    -e "s#__Y2__#${Y2}#g" \
    -e "s#__Z2__#${Z2}#g" \
    -e "s#__cm__#${CM2}#g" \
    "$TEMPLATE_NAC" > "temp/run_${count}_nac.mac"
  Gate "temp/run_${count}_nac.mac"

  # AC
  sed \
    -e "s#__OUTPUT_NAME__#output/AC/img_${count}#g" \
    -e "s#__R1__#${R1}#g" \
    -e "s#__SR1__#${SR1}#g" \
    -e "s#__X1__#${X1}#g" \
    -e "s#__Y1__#${Y1}#g" \
    -e "s#__Z1__#${Z1}#g" \
    -e "s#__cm__#${CM1}#g" \
    -e "s#__R2__#${R2}#g" \
    -e "s#__SR2__#${SR2}#g" \
    -e "s#__X2__#${X2}#g" \
    -e "s#__Y2__#${Y2}#g" \
    -e "s#__Z2__#${Z2}#g" \
    -e "s#__cm__#${CM2}#g" \
    "$TEMPLATE_AC" > "temp/run_${count}_ac.mac"
  Gate "temp/run_${count}_ac.mac"

  echo "✓ Created & ran temp/run_${count}_*.mac"
  count=$((count + 1))
done < parameters_double.txt

echo "✅ Ολοκληρώθηκαν όλα τα runs."

