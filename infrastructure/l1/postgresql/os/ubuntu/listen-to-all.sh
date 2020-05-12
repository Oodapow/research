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

ipv4str="host    all             all              0.0.0.0/0                       md5"
ipv6str="host    all             all              ::/0                            md5"

if grep -q ipv4str $2 ; then
    echo "Correct IPv4 settings found, doing nothing"
fi

if ! grep -q ipv4str $2 ; then
    echo "No IPv4 settings found, adding one at the end"
    echo ipv4str >> $2
fi

if grep -q ipv6str $2 ; then
    echo "Correct IPv6 settings found, doing nothing"
fi

if ! grep -q ipv6str $2 ; then
    echo "No IPv6 settings found, adding one at the end"
    echo ipv6str >> $2
fi