if grep -q "^listen_addresses = '\*'" $1 ; then
    echo "Correct listen_addresses found, doing nothing"
    exit
fi

if ! grep -q "^listen_addresses =.*" $1 ; then
    echo "No listen_addresses found, adding one at the end"
    echo "listen_addresses = '*'" >> $1
    exit
fi

if grep -q "^listen_addresses =.*" $1 ; then
    echo "Wrong listen_addresses found, commenting them out"
    sed -i "s/^\(listen_addresses.*\)/#\1 Commented out by Name YYYY-MM-DD/" $1

    echo "Adding correct one"
    sed -i "/^#listen_addresses/a listen_addresses = '\*'" $1
fi

pg_hba=$(find / -name "pg_hba.conf" -print -quit)
echo "host    all             all              0.0.0.0/0                       md5" >> pg_hba
echo "host    all             all              ::/0                            md5" >> pg_hba
printf "$pg_hba edited."