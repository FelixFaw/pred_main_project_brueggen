#bereinigen

Done    #Dauer Arbeitszeit runden auf .0 (59.9999 auf 60.0)
Done    #Menge gesamt = zu Menge gesamt NIO benennen (einfach umbenennen)
to check     #alle zeilen raus wo bemerkeung + OP den wert null (dann auch kein Ausfall i guess)
unvollständig    #Menge N.io + Menge I.O L4 + Menge I.O L5 von null-Wert  auf 0 (Gedanke: lstm besser mit 0 als mit Null weil wir hier Mengen als Input haben)
unvollständig    # bei allen Dauer Anlagen Ausfall von Null-Wert auf 0

fehlt #entfernen : fehlercode (nur einmal was drinnen stehend)

fehlt #syn daten:

    #Zeit in Stunden nach Schichtbeginn
    #Tag, Monat, Woche, Quartal, Jahr
    #Wochenende , kein Wochenende
    # Monatsanfang/-ende
    #Zeit seit letztem Fehler GLOBAL
    #Zeit seit letzter Wartung GLOBAL
    # boolean Ausfall/kein Ausfall

    # Wie lange bis die Maschine erneut ausfällt (Maschine fällt nach 30 Minuten aus, danach nach 40 Minuten etc... ) STATIONBEZOGEN
    # durchschnittliche Zeitspanne je Ausfall einer einzelnen Maschine (bps. im durchschnitt alle 30 Minuten fällt Maschine XY aus) STATIONBEZOGEN
    #wie viele produkte wurden angefertigt seit letztem Ausfall GLOBAL
    #wie viele Produkte NIO seit letzten Ausfall
    #gleitende Mittelwerte der Fehlerhäufigkeit (Rolling Mean über zb 7 oder 30 Tage, ich würde vmtl auf woche UND monat das ausgeben einmal)


    #Idee : Sequenzmuster aus der Maschinenreihenfolge erstellen
    #letzte 5 Maschinen vor der vom Ausfall betroffenen Maschine
    #nächsten 5 Maschinen vor der vom Ausfall betroffenen Maschine
    
    #letzten 5 Maschinen-Ausfälle vor der vom Ausfall betroffenen Maschine
    #letzten 5 Maschinen-Ausfälle vor der vom Ausfall betroffenen Maschine ( PRÜFEN:kann man das input ins lstm geben)

    #ÜBERTRAG DER KUNDENAUFTRAGSNUMMER aus prozessliste
