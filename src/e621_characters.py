from typing import Dict, List, Set

"""
This file is used in the Illustrious custom node for ComfyUI.
It contains a dictionary of characters with their associated triggers and core tags.
"""

E621_CHARACTERS: Dict[str, Dict[str, List[str]]] = {
    "fan_character": {
        "character": ["fan_character"],
        "trigger": ["fan character, hasbro"],
    },
    "twilight_sparkle_(mlp)": {
        "character": ["twilight_sparkle_(mlp)"],
        "trigger": ["twilight sparkle \\(mlp\\), my little pony"],
    },
    "judy_hopps": {"character": ["judy_hopps"], "trigger": ["judy hopps, disney"]},
    "fluttershy_(mlp)": {
        "character": ["fluttershy_(mlp)"],
        "trigger": ["fluttershy \\(mlp\\), my little pony"],
    },
    "nick_wilde": {"character": ["nick_wilde"], "trigger": ["nick wilde, disney"]},
    "rainbow_dash_(mlp)": {
        "character": ["rainbow_dash_(mlp)"],
        "trigger": ["rainbow dash \\(mlp\\), my little pony"],
    },
    "rarity_(mlp)": {
        "character": ["rarity_(mlp)"],
        "trigger": ["rarity \\(mlp\\), my little pony"],
    },
    "pinkie_pie_(mlp)": {
        "character": ["pinkie_pie_(mlp)"],
        "trigger": ["pinkie pie \\(mlp\\), my little pony"],
    },
    "rouge_the_bat": {
        "character": ["rouge_the_bat"],
        "trigger": ["rouge the bat, sonic the hedgehog \\(series\\)"],
    },
    "applejack_(mlp)": {
        "character": ["applejack_(mlp)"],
        "trigger": ["applejack \\(mlp\\), my little pony"],
    },
    "princess_celestia_(mlp)": {
        "character": ["princess_celestia_(mlp)"],
        "trigger": ["princess celestia \\(mlp\\), my little pony"],
    },
    "princess_luna_(mlp)": {
        "character": ["princess_luna_(mlp)"],
        "trigger": ["princess luna \\(mlp\\), my little pony"],
    },
    "sonic_the_hedgehog": {
        "character": ["sonic_the_hedgehog"],
        "trigger": ["sonic the hedgehog, sonic the hedgehog \\(series\\)"],
    },
    "miles_prower": {
        "character": ["miles_prower"],
        "trigger": ["miles prower, sonic the hedgehog \\(series\\)"],
    },
    "amy_rose": {
        "character": ["amy_rose"],
        "trigger": ["amy rose, sonic the hedgehog \\(series\\)"],
    },
    "loona_(helluva_boss)": {
        "character": ["loona_(helluva_boss)"],
        "trigger": ["loona \\(helluva boss\\), helluva boss"],
    },
    "krystal_(star_fox)": {
        "character": ["krystal_(star_fox)"],
        "trigger": ["krystal \\(star fox\\), star fox"],
    },
    "toriel": {"character": ["toriel"], "trigger": ["toriel, undertale \\(series\\)"]},
    "isabelle_(animal_crossing)": {
        "character": ["isabelle_(animal_crossing)"],
        "trigger": ["isabelle \\(animal crossing\\), animal crossing"],
    },
    "bowser": {"character": ["bowser"], "trigger": ["bowser, mario bros"]},
    "spike_(mlp)": {
        "character": ["spike_(mlp)"],
        "trigger": ["spike \\(mlp\\), my little pony"],
    },
    "fox_mccloud": {"character": ["fox_mccloud"], "trigger": ["fox mccloud, star fox"]},
    "queen_chrysalis_(mlp)": {
        "character": ["queen_chrysalis_(mlp)"],
        "trigger": ["queen chrysalis \\(mlp\\), my little pony"],
    },
    "anon": {"character": ["anon"], "trigger": ["anon, hasbro"]},
    "sweetie_belle_(mlp)": {
        "character": ["sweetie_belle_(mlp)"],
        "trigger": ["sweetie belle \\(mlp\\), my little pony"],
    },
    "blaze_the_cat": {
        "character": ["blaze_the_cat"],
        "trigger": ["blaze the cat, sonic the hedgehog \\(series\\)"],
    },
    "trixie_(mlp)": {
        "character": ["trixie_(mlp)"],
        "trigger": ["trixie \\(mlp\\), my little pony"],
    },
    "princess_cadance_(mlp)": {
        "character": ["princess_cadance_(mlp)"],
        "trigger": ["princess cadance \\(mlp\\), my little pony"],
    },
    "ankha_(animal_crossing)": {
        "character": ["ankha_(animal_crossing)"],
        "trigger": ["ankha \\(animal crossing\\), animal crossing"],
    },
    "foxy_(fnaf)": {
        "character": ["foxy_(fnaf)"],
        "trigger": ["foxy \\(fnaf\\), scottgames"],
    },
    "shadow_the_hedgehog": {
        "character": ["shadow_the_hedgehog"],
        "trigger": ["shadow the hedgehog, sonic the hedgehog \\(series\\)"],
    },
    "cream_the_rabbit": {
        "character": ["cream_the_rabbit"],
        "trigger": ["cream the rabbit, sonic the hedgehog \\(series\\)"],
    },
    "big_macintosh_(mlp)": {
        "character": ["big_macintosh_(mlp)"],
        "trigger": ["big macintosh \\(mlp\\), my little pony"],
    },
    "shining_armor_(mlp)": {
        "character": ["shining_armor_(mlp)"],
        "trigger": ["shining armor \\(mlp\\), my little pony"],
    },
    "asriel_dreemurr": {
        "character": ["asriel_dreemurr"],
        "trigger": ["asriel dreemurr, undertale \\(series\\)"],
    },
    "apple_bloom_(mlp)": {
        "character": ["apple_bloom_(mlp)"],
        "trigger": ["apple bloom \\(mlp\\), my little pony"],
    },
    "legoshi_(beastars)": {
        "character": ["legoshi_(beastars)"],
        "trigger": ["legoshi \\(beastars\\), beastars"],
    },
    "scootaloo_(mlp)": {
        "character": ["scootaloo_(mlp)"],
        "trigger": ["scootaloo \\(mlp\\), my little pony"],
    },
    "stitch_(lilo_and_stitch)": {
        "character": ["stitch_(lilo_and_stitch)"],
        "trigger": ["stitch \\(lilo and stitch\\), disney"],
    },
    "midna": {"character": ["midna"], "trigger": ["midna, twilight princess"]},
    "link": {"character": ["link"], "trigger": ["link, the legend of zelda"]},
    "nicole_watterson": {
        "character": ["nicole_watterson"],
        "trigger": ["nicole watterson, cartoon network"],
    },
    "derpy_hooves_(mlp)": {
        "character": ["derpy_hooves_(mlp)"],
        "trigger": ["derpy hooves \\(mlp\\), my little pony"],
    },
    "frisk_(undertale)": {
        "character": ["frisk_(undertale)"],
        "trigger": ["frisk \\(undertale\\), undertale \\(series\\)"],
    },
    "falco_lombardi": {
        "character": ["falco_lombardi"],
        "trigger": ["falco lombardi, star fox"],
    },
    "ralsei": {"character": ["ralsei"], "trigger": ["ralsei, undertale \\(series\\)"]},
    "undyne": {"character": ["undyne"], "trigger": ["undyne, undertale \\(series\\)"]},
    "sally_acorn": {
        "character": ["sally_acorn"],
        "trigger": ["sally acorn, sonic the hedgehog \\(series\\)"],
    },
    "wolf_o'donnell": {
        "character": ["wolf_o'donnell"],
        "trigger": ["wolf o'donnell, star fox"],
    },
    "pokemon_trainer": {
        "character": ["pokemon_trainer"],
        "trigger": ["pokemon trainer, pokemon"],
    },
    "coco_bandicoot": {
        "character": ["coco_bandicoot"],
        "trigger": ["coco bandicoot, crash bandicoot \\(series\\)"],
    },
    "vanilla_the_rabbit": {
        "character": ["vanilla_the_rabbit"],
        "trigger": ["vanilla the rabbit, sonic the hedgehog \\(series\\)"],
    },
    "roxanne_wolf": {
        "character": ["roxanne_wolf"],
        "trigger": ["roxanne wolf, five nights at freddy's: security breach"],
    },
    "susie_(deltarune)": {
        "character": ["susie_(deltarune)"],
        "trigger": ["susie \\(deltarune\\), undertale \\(series\\)"],
    },
    "knuckles_the_echidna": {
        "character": ["knuckles_the_echidna"],
        "trigger": ["knuckles the echidna, sonic the hedgehog \\(series\\)"],
    },
    "wonderbolts_(mlp)": {
        "character": ["wonderbolts_(mlp)"],
        "trigger": ["wonderbolts \\(mlp\\), my little pony"],
    },
    "alphys": {"character": ["alphys"], "trigger": ["alphys, undertale \\(series\\)"]},
    "vinyl_scratch_(mlp)": {
        "character": ["vinyl_scratch_(mlp)"],
        "trigger": ["vinyl scratch \\(mlp\\), my little pony"],
    },
    "lola_bunny": {
        "character": ["lola_bunny"],
        "trigger": ["lola bunny, warner brothers"],
    },
    "toy_chica_(fnaf)": {
        "character": ["toy_chica_(fnaf)"],
        "trigger": ["toy chica \\(fnaf\\), five nights at freddy's 2"],
    },
    "spyro": {"character": ["spyro"], "trigger": ["spyro, spyro the dragon"]},
    "sans_(undertale)": {
        "character": ["sans_(undertale)"],
        "trigger": ["sans \\(undertale\\), undertale \\(series\\)"],
    },
    "bonnie_(fnaf)": {
        "character": ["bonnie_(fnaf)"],
        "trigger": ["bonnie \\(fnaf\\), five nights at freddy's"],
    },
    "discord_(mlp)": {
        "character": ["discord_(mlp)"],
        "trigger": ["discord \\(mlp\\), my little pony"],
    },
    "starlight_glimmer_(mlp)": {
        "character": ["starlight_glimmer_(mlp)"],
        "trigger": ["starlight glimmer \\(mlp\\), my little pony"],
    },
    "asgore_dreemurr": {
        "character": ["asgore_dreemurr"],
        "trigger": ["asgore dreemurr, undertale \\(series\\)"],
    },
    "kris_(deltarune)": {
        "character": ["kris_(deltarune)"],
        "trigger": ["kris \\(deltarune\\), undertale \\(series\\)"],
    },
    "unknown_character": {
        "character": ["unknown_character"],
        "trigger": ["unknown character, mythology"],
    },
    "octavia_(mlp)": {
        "character": ["octavia_(mlp)"],
        "trigger": ["octavia \\(mlp\\), my little pony"],
    },
    "master_tigress": {
        "character": ["master_tigress"],
        "trigger": ["master tigress, kung fu panda"],
    },
    "death_(puss_in_boots)": {
        "character": ["death_(puss_in_boots)"],
        "trigger": ["death \\(puss in boots\\), puss in boots \\(dreamworks\\)"],
    },
    "gumball_watterson": {
        "character": ["gumball_watterson"],
        "trigger": ["gumball watterson, cartoon network"],
    },
    "freddy_(fnaf)": {
        "character": ["freddy_(fnaf)"],
        "trigger": ["freddy \\(fnaf\\), scottgames"],
    },
    "kindred_(lol)": {
        "character": ["kindred_(lol)"],
        "trigger": ["kindred \\(lol\\), riot games"],
    },
    "nightmare_moon_(mlp)": {
        "character": ["nightmare_moon_(mlp)"],
        "trigger": ["nightmare moon \\(mlp\\), my little pony"],
    },
    "lamb_(lol)": {
        "character": ["lamb_(lol)"],
        "trigger": ["lamb \\(lol\\), riot games"],
    },
    "diane_foxington": {
        "character": ["diane_foxington"],
        "trigger": ["diane foxington, the bad guys"],
    },
    "villager_(animal_crossing)": {
        "character": ["villager_(animal_crossing)"],
        "trigger": ["villager \\(animal crossing\\), animal crossing"],
    },
    "lyra_heartstrings_(mlp)": {
        "character": ["lyra_heartstrings_(mlp)"],
        "trigger": ["lyra heartstrings \\(mlp\\), my little pony"],
    },
    "princess_zelda": {
        "character": ["princess_zelda"],
        "trigger": ["princess zelda, the legend of zelda"],
    },
    "rivet_(ratchet_and_clank)": {
        "character": ["rivet_(ratchet_and_clank)"],
        "trigger": ["rivet \\(ratchet and clank\\), sony corporation"],
    },
    "asriel_dreemurr_(god_form)": {
        "character": ["asriel_dreemurr_(god_form)"],
        "trigger": ["asriel dreemurr \\(god form\\), undertale \\(series\\)"],
    },
    "scp-1471-a": {
        "character": ["scp-1471-a"],
        "trigger": ["scp-1471-a, scp foundation"],
    },
    "scp-1471": {"character": ["scp-1471"], "trigger": ["scp-1471, scp foundation"]},
    "chica_(fnaf)": {
        "character": ["chica_(fnaf)"],
        "trigger": ["chica \\(fnaf\\), five nights at freddy's"],
    },
    "princess_peach": {
        "character": ["princess_peach"],
        "trigger": ["princess peach, mario bros"],
    },
    "mae_borowski": {
        "character": ["mae_borowski"],
        "trigger": ["mae borowski, night in the woods"],
    },
    "mangle_(fnaf)": {
        "character": ["mangle_(fnaf)"],
        "trigger": ["mangle \\(fnaf\\), five nights at freddy's 2"],
    },
    "mario": {"character": ["mario"], "trigger": ["mario, mario bros"]},
    "anubis": {"character": ["anubis"], "trigger": ["anubis, egyptian mythology"]},
    "moxxie_(helluva_boss)": {
        "character": ["moxxie_(helluva_boss)"],
        "trigger": ["moxxie \\(helluva boss\\), helluva boss"],
    },
    "rocket_raccoon": {
        "character": ["rocket_raccoon"],
        "trigger": ["rocket raccoon, marvel"],
    },
    "michiru_kagemori": {
        "character": ["michiru_kagemori"],
        "trigger": ["michiru kagemori, studio trigger"],
    },
    "tom_nook_(animal_crossing)": {
        "character": ["tom_nook_(animal_crossing)"],
        "trigger": ["tom nook \\(animal crossing\\), animal crossing"],
    },
    "carmelita_fox": {
        "character": ["carmelita_fox"],
        "trigger": ["carmelita fox, sucker punch productions"],
    },
    "temmie_(undertale)": {
        "character": ["temmie_(undertale)"],
        "trigger": ["temmie \\(undertale\\), undertale \\(series\\)"],
    },
    "sunset_shimmer_(eg)": {
        "character": ["sunset_shimmer_(eg)"],
        "trigger": ["sunset shimmer \\(eg\\), equestria girls"],
    },
    "muffet": {"character": ["muffet"], "trigger": ["muffet, undertale \\(series\\)"]},
    "stolas_(helluva_boss)": {
        "character": ["stolas_(helluva_boss)"],
        "trigger": ["stolas \\(helluva boss\\), helluva boss"],
    },
    "spitfire_(mlp)": {
        "character": ["spitfire_(mlp)"],
        "trigger": ["spitfire \\(mlp\\), my little pony"],
    },
    "toy_bonnie_(fnaf)": {
        "character": ["toy_bonnie_(fnaf)"],
        "trigger": ["toy bonnie \\(fnaf\\), five nights at freddy's 2"],
    },
    "katia_managan": {
        "character": ["katia_managan"],
        "trigger": ["katia managan, the elder scrolls"],
    },
    "pinkamena_(mlp)": {
        "character": ["pinkamena_(mlp)"],
        "trigger": ["pinkamena \\(mlp\\), my little pony"],
    },
    "maid_marian": {"character": ["maid_marian"], "trigger": ["maid marian, disney"]},
    "gilda_(mlp)": {
        "character": ["gilda_(mlp)"],
        "trigger": ["gilda \\(mlp\\), my little pony"],
    },
    "gadget_hackwrench": {
        "character": ["gadget_hackwrench"],
        "trigger": ["gadget hackwrench, disney"],
    },
    "king_sombra_(mlp)": {
        "character": ["king_sombra_(mlp)"],
        "trigger": ["king sombra \\(mlp\\), my little pony"],
    },
    "kirby": {"character": ["kirby"], "trigger": ["kirby, kirby \\(series\\)"]},
    "simba_(the_lion_king)": {
        "character": ["simba_(the_lion_king)"],
        "trigger": ["simba \\(the lion king\\), disney"],
    },
    "millie_(helluva_boss)": {
        "character": ["millie_(helluva_boss)"],
        "trigger": ["millie \\(helluva boss\\), helluva boss"],
    },
    "nala_(the_lion_king)": {
        "character": ["nala_(the_lion_king)"],
        "trigger": ["nala \\(the lion king\\), disney"],
    },
    "mr._wolf_(the_bad_guys)": {
        "character": ["mr._wolf_(the_bad_guys)"],
        "trigger": ["mr. wolf \\(the bad guys\\), the bad guys"],
    },
    "noelle_holiday": {
        "character": ["noelle_holiday"],
        "trigger": ["noelle holiday, undertale \\(series\\)"],
    },
    "sheriff_mao_mao_mao": {
        "character": ["sheriff_mao_mao_mao"],
        "trigger": ["sheriff mao mao mao, mao mao: heroes of pure heart"],
    },
    "haida_(aggretsuko)": {
        "character": ["haida_(aggretsuko)"],
        "trigger": ["haida \\(aggretsuko\\), sanrio"],
    },
    "koopaling": {"character": ["koopaling"], "trigger": ["koopaling, mario bros"]},
    "finnick_(zootopia)": {
        "character": ["finnick_(zootopia)"],
        "trigger": ["finnick \\(zootopia\\), disney"],
    },
    "cynder": {"character": ["cynder"], "trigger": ["cynder, spyro the dragon"]},
    "zecora_(mlp)": {
        "character": ["zecora_(mlp)"],
        "trigger": ["zecora \\(mlp\\), hasbro"],
    },
    "fang_(gvh)": {
        "character": ["fang_(gvh)"],
        "trigger": ["fang \\(gvh\\), goodbye volcano high"],
    },
    "felicia_(darkstalkers)": {
        "character": ["felicia_(darkstalkers)"],
        "trigger": ["felicia \\(darkstalkers\\), darkstalkers"],
    },
    "papyrus_(undertale)": {
        "character": ["papyrus_(undertale)"],
        "trigger": ["papyrus \\(undertale\\), undertale \\(series\\)"],
    },
    "lamb_(cult_of_the_lamb)": {
        "character": ["lamb_(cult_of_the_lamb)"],
        "trigger": ["lamb \\(cult of the lamb\\), cult of the lamb"],
    },
    "retsuko": {"character": ["retsuko"], "trigger": ["retsuko, sanrio"]},
    "shantae": {"character": ["shantae"], "trigger": ["shantae, wayforward"]},
    "ratchet": {"character": ["ratchet"], "trigger": ["ratchet, sony corporation"]},
    "haru_(beastars)": {
        "character": ["haru_(beastars)"],
        "trigger": ["haru \\(beastars\\), beastars"],
    },
    "juno_(beastars)": {
        "character": ["juno_(beastars)"],
        "trigger": ["juno \\(beastars\\), beastars"],
    },
    "cutie_mark_crusaders_(mlp)": {
        "character": ["cutie_mark_crusaders_(mlp)"],
        "trigger": ["cutie mark crusaders \\(mlp\\), my little pony"],
    },
    "louis_(beastars)": {
        "character": ["louis_(beastars)"],
        "trigger": ["louis \\(beastars\\), beastars"],
    },
    "toothless": {
        "character": ["toothless"],
        "trigger": ["toothless, how to train your dragon"],
    },
    "blitzo_(helluva_boss)": {
        "character": ["blitzo_(helluva_boss)"],
        "trigger": ["blitzo \\(helluva boss\\), helluva boss"],
    },
    "samus_aran": {"character": ["samus_aran"], "trigger": ["samus aran, nintendo"]},
    "crash_bandicoot": {
        "character": ["crash_bandicoot"],
        "trigger": ["crash bandicoot, crash bandicoot \\(series\\)"],
    },
    "kitty_katswell": {
        "character": ["kitty_katswell"],
        "trigger": ["kitty katswell, t.u.f.f. puppy"],
    },
    "glamrock_freddy": {
        "character": ["glamrock_freddy"],
        "trigger": ["glamrock freddy, five nights at freddy's: security breach"],
    },
    "soarin_(mlp)": {
        "character": ["soarin_(mlp)"],
        "trigger": ["soarin \\(mlp\\), my little pony"],
    },
    "poppy_(lol)": {
        "character": ["poppy_(lol)"],
        "trigger": ["poppy \\(lol\\), riot games"],
    },
    "klonoa": {"character": ["klonoa"], "trigger": ["klonoa, bandai namco"]},
    "tawna_bandicoot": {
        "character": ["tawna_bandicoot"],
        "trigger": ["tawna bandicoot, crash bandicoot \\(series\\)"],
    },
    "jenny_wakeman": {
        "character": ["jenny_wakeman"],
        "trigger": ["jenny wakeman, my life as a teenage robot"],
    },
    "kass_(tloz)": {
        "character": ["kass_(tloz)"],
        "trigger": ["kass \\(tloz\\), breath of the wild"],
    },
    "maud_pie_(mlp)": {
        "character": ["maud_pie_(mlp)"],
        "trigger": ["maud pie \\(mlp\\), my little pony"],
    },
    "tristana_(lol)": {
        "character": ["tristana_(lol)"],
        "trigger": ["tristana \\(lol\\), riot games"],
    },
    "jack_savage": {"character": ["jack_savage"], "trigger": ["jack savage, disney"]},
    "bugs_bunny": {
        "character": ["bugs_bunny"],
        "trigger": ["bugs bunny, looney tunes"],
    },
    "ych_(character)": {
        "character": ["ych_(character)"],
        "trigger": ["ych \\(character\\), mythology"],
    },
    "silver_the_hedgehog": {
        "character": ["silver_the_hedgehog"],
        "trigger": ["silver the hedgehog, sonic the hedgehog \\(series\\)"],
    },
    "peg_pete": {"character": ["peg_pete"], "trigger": ["peg pete, disney"]},
    "fenneko": {"character": ["fenneko"], "trigger": ["fenneko, sanrio"]},
    "princess_ember_(mlp)": {
        "character": ["princess_ember_(mlp)"],
        "trigger": ["princess ember \\(mlp\\), my little pony"],
    },
    "hornet_(hollow_knight)": {
        "character": ["hornet_(hollow_knight)"],
        "trigger": ["hornet \\(hollow knight\\), team cherry"],
    },
    "silver_soul": {"character": ["silver_soul"], "trigger": ["silver soul, pokemon"]},
    "marina_(splatoon)": {
        "character": ["marina_(splatoon)"],
        "trigger": ["marina \\(splatoon\\), splatoon"],
    },
    "chara_(undertale)": {
        "character": ["chara_(undertale)"],
        "trigger": ["chara \\(undertale\\), undertale \\(series\\)"],
    },
    "bea_santello": {
        "character": ["bea_santello"],
        "trigger": ["bea santello, night in the woods"],
    },
    "meow_skulls_(fortnite)": {
        "character": ["meow_skulls_(fortnite)"],
        "trigger": ["meow skulls \\(fortnite\\), fortnite"],
    },
    "minerva_mink": {
        "character": ["minerva_mink"],
        "trigger": ["minerva mink, warner brothers"],
    },
    "godzilla": {
        "character": ["godzilla"],
        "trigger": ["godzilla, godzilla \\(series\\)"],
    },
    "morgana_(persona)": {
        "character": ["morgana_(persona)"],
        "trigger": ["morgana \\(persona\\), persona \\(series\\)"],
    },
    "sly_cooper": {
        "character": ["sly_cooper"],
        "trigger": ["sly cooper, sucker punch productions"],
    },
    "puss_in_boots_(character)": {
        "character": ["puss_in_boots_(character)"],
        "trigger": ["puss in boots \\(character\\), puss in boots \\(dreamworks\\)"],
    },
    "chief_bogo": {"character": ["chief_bogo"], "trigger": ["chief bogo, disney"]},
    "freya_crescent": {
        "character": ["freya_crescent"],
        "trigger": ["freya crescent, square enix"],
    },
    "raphael_(tmnt)": {
        "character": ["raphael_(tmnt)"],
        "trigger": ["raphael \\(tmnt\\), teenage mutant ninja turtles"],
    },
    "springtrap_(fnaf)": {
        "character": ["springtrap_(fnaf)"],
        "trigger": ["springtrap \\(fnaf\\), scottgames"],
    },
    "octavia_(helluva_boss)": {
        "character": ["octavia_(helluva_boss)"],
        "trigger": ["octavia \\(helluva boss\\), helluva boss"],
    },
    "fifi_la_fume": {
        "character": ["fifi_la_fume"],
        "trigger": ["fifi la fume, warner brothers"],
    },
    "protagonist_(tas)": {
        "character": ["protagonist_(tas)"],
        "trigger": ["protagonist \\(tas\\), lifewonders"],
    },
    "amaterasu_(okami)": {
        "character": ["amaterasu_(okami)"],
        "trigger": ["amaterasu \\(okami\\), capcom"],
    },
    "whisper_the_wolf": {
        "character": ["whisper_the_wolf"],
        "trigger": ["whisper the wolf, sonic the hedgehog \\(series\\)"],
    },
    "background_character": {
        "character": ["background_character"],
        "trigger": ["background character, mythology"],
    },
    "rumi_usagiyama": {
        "character": ["rumi_usagiyama"],
        "trigger": ["rumi usagiyama, my hero academia"],
    },
    "leonardo_(tmnt)": {
        "character": ["leonardo_(tmnt)"],
        "trigger": ["leonardo \\(tmnt\\), teenage mutant ninja turtles"],
    },
    "ahri_(lol)": {
        "character": ["ahri_(lol)"],
        "trigger": ["ahri \\(lol\\), riot games"],
    },
    "bonbon_(mlp)": {
        "character": ["bonbon_(mlp)"],
        "trigger": ["bonbon \\(mlp\\), my little pony"],
    },
    "rigby_(regular_show)": {
        "character": ["rigby_(regular_show)"],
        "trigger": ["rigby \\(regular show\\), cartoon network"],
    },
    "red_crown_(cult_of_the_lamb)": {
        "character": ["red_crown_(cult_of_the_lamb)"],
        "trigger": ["red crown \\(cult of the lamb\\), cult of the lamb"],
    },
    "link_(wolf_form)": {
        "character": ["link_(wolf_form)"],
        "trigger": ["link \\(wolf form\\), the legend of zelda"],
    },
    "royal_guard_(mlp)": {
        "character": ["royal_guard_(mlp)"],
        "trigger": ["royal guard \\(mlp\\), my little pony"],
    },
    "bunnie_rabbot": {
        "character": ["bunnie_rabbot"],
        "trigger": ["bunnie rabbot, sonic the hedgehog \\(series\\)"],
    },
    "twilight_velvet_(mlp)": {
        "character": ["twilight_velvet_(mlp)"],
        "trigger": ["twilight velvet \\(mlp\\), my little pony"],
    },
    "beerus": {"character": ["beerus"], "trigger": ["beerus, dragon ball"]},
    "mordecai_(regular_show)": {
        "character": ["mordecai_(regular_show)"],
        "trigger": ["mordecai \\(regular show\\), cartoon network"],
    },
    "anon_(snoot_game)": {
        "character": ["anon_(snoot_game)"],
        "trigger": ["anon \\(snoot game\\), cavemanon studios"],
    },
    "cheerilee_(mlp)": {
        "character": ["cheerilee_(mlp)"],
        "trigger": ["cheerilee \\(mlp\\), my little pony"],
    },
    "flutterbat_(mlp)": {
        "character": ["flutterbat_(mlp)"],
        "trigger": ["flutterbat \\(mlp\\), my little pony"],
    },
    "puro_(changed)": {
        "character": ["puro_(changed)"],
        "trigger": ["puro \\(changed\\), changed \\(video game\\)"],
    },
    "raymond_(animal_crossing)": {
        "character": ["raymond_(animal_crossing)"],
        "trigger": ["raymond \\(animal crossing\\), animal crossing"],
    },
    "montgomery_gator": {
        "character": ["montgomery_gator"],
        "trigger": ["montgomery gator, five nights at freddy's: security breach"],
    },
    "moritaka_(tas)": {
        "character": ["moritaka_(tas)"],
        "trigger": ["moritaka \\(tas\\), lifewonders"],
    },
    "glamrock_chica": {
        "character": ["glamrock_chica"],
        "trigger": ["glamrock chica, five nights at freddy's: security breach"],
    },
    "warfare_krystal": {
        "character": ["warfare_krystal"],
        "trigger": ["warfare krystal, star fox"],
    },
    "teemo_(lol)": {
        "character": ["teemo_(lol)"],
        "trigger": ["teemo \\(lol\\), riot games"],
    },
    "sticks_the_jungle_badger": {
        "character": ["sticks_the_jungle_badger"],
        "trigger": ["sticks the jungle badger, sonic the hedgehog \\(series\\)"],
    },
    "tasque_manager": {
        "character": ["tasque_manager"],
        "trigger": ["tasque manager, undertale \\(series\\)"],
    },
    "master_po_ping": {
        "character": ["master_po_ping"],
        "trigger": ["master po ping, kung fu panda"],
    },
    "pikachu_libre": {
        "character": ["pikachu_libre"],
        "trigger": ["pikachu libre, pokemon"],
    },
    "robin_hood": {"character": ["robin_hood"], "trigger": ["robin hood, disney"]},
    "leib_(tas)": {
        "character": ["leib_(tas)"],
        "trigger": ["leib \\(tas\\), lifewonders"],
    },
    "bandit_heeler": {
        "character": ["bandit_heeler"],
        "trigger": ["bandit heeler, bluey \\(series\\)"],
    },
    "wave_the_swallow": {
        "character": ["wave_the_swallow"],
        "trigger": ["wave the swallow, sonic the hedgehog \\(series\\)"],
    },
    "elora": {"character": ["elora"], "trigger": ["elora, spyro the dragon"]},
    "stella_(helluva_boss)": {
        "character": ["stella_(helluva_boss)"],
        "trigger": ["stella \\(helluva boss\\), helluva boss"],
    },
    "flowey_the_flower": {
        "character": ["flowey_the_flower"],
        "trigger": ["flowey the flower, undertale \\(series\\)"],
    },
    "skye_(zootopia)": {
        "character": ["skye_(zootopia)"],
        "trigger": ["skye \\(zootopia\\), disney"],
    },
    "sandy_cheeks": {
        "character": ["sandy_cheeks"],
        "trigger": ["sandy cheeks, spongebob squarepants"],
    },
    "braeburn_(mlp)": {
        "character": ["braeburn_(mlp)"],
        "trigger": ["braeburn \\(mlp\\), my little pony"],
    },
    "banjo_(banjo-kazooie)": {
        "character": ["banjo_(banjo-kazooie)"],
        "trigger": ["banjo \\(banjo-kazooie\\), rareware"],
    },
    "tempest_shadow_(mlp)": {
        "character": ["tempest_shadow_(mlp)"],
        "trigger": ["tempest shadow \\(mlp\\), my little pony"],
    },
    "digby_(animal_crossing)": {
        "character": ["digby_(animal_crossing)"],
        "trigger": ["digby \\(animal crossing\\), animal crossing"],
    },
    "badgerclops": {
        "character": ["badgerclops"],
        "trigger": ["badgerclops, mao mao: heroes of pure heart"],
    },
    "blaidd_(elden_ring)": {
        "character": ["blaidd_(elden_ring)"],
        "trigger": ["blaidd \\(elden ring\\), fromsoftware"],
    },
    "smolder_(mlp)": {
        "character": ["smolder_(mlp)"],
        "trigger": ["smolder \\(mlp\\), my little pony"],
    },
    "michelangelo_(tmnt)": {
        "character": ["michelangelo_(tmnt)"],
        "trigger": ["michelangelo \\(tmnt\\), teenage mutant ninja turtles"],
    },
    "jack_(beastars)": {
        "character": ["jack_(beastars)"],
        "trigger": ["jack \\(beastars\\), beastars"],
    },
    "gideon_grey": {"character": ["gideon_grey"], "trigger": ["gideon grey, disney"]},
    "callie_briggs": {
        "character": ["callie_briggs"],
        "trigger": ["callie briggs, swat kats"],
    },
    "shirou_ogami": {
        "character": ["shirou_ogami"],
        "trigger": ["shirou ogami, studio trigger"],
    },
    "flick_(animal_crossing)": {
        "character": ["flick_(animal_crossing)"],
        "trigger": ["flick \\(animal crossing\\), animal crossing"],
    },
    "flora_(twokinds)": {
        "character": ["flora_(twokinds)"],
        "trigger": ["flora \\(twokinds\\), twokinds"],
    },
    "surge_the_tenrec": {
        "character": ["surge_the_tenrec"],
        "trigger": ["surge the tenrec, sonic the hedgehog \\(series\\)"],
    },
    "marble_pie_(mlp)": {
        "character": ["marble_pie_(mlp)"],
        "trigger": ["marble pie \\(mlp\\), my little pony"],
    },
    "waaifu_(arknights)": {
        "character": ["waaifu_(arknights)"],
        "trigger": ["waaifu \\(arknights\\), studio montagne"],
    },
    "tikal_the_echidna": {
        "character": ["tikal_the_echidna"],
        "trigger": ["tikal the echidna, sonic the hedgehog \\(series\\)"],
    },
    "marie_(splatoon)": {
        "character": ["marie_(splatoon)"],
        "trigger": ["marie \\(splatoon\\), splatoon"],
    },
    "donatello_(tmnt)": {
        "character": ["donatello_(tmnt)"],
        "trigger": ["donatello \\(tmnt\\), teenage mutant ninja turtles"],
    },
    "yasuyori_(tas)": {
        "character": ["yasuyori_(tas)"],
        "trigger": ["yasuyori \\(tas\\), lifewonders"],
    },
    "horkeu_kamui_(tas)": {
        "character": ["horkeu_kamui_(tas)"],
        "trigger": ["horkeu kamui \\(tas\\), lifewonders"],
    },
    "audie_(animal_crossing)": {
        "character": ["audie_(animal_crossing)"],
        "trigger": ["audie \\(animal crossing\\), animal crossing"],
    },
    "tangle_the_lemur": {
        "character": ["tangle_the_lemur"],
        "trigger": ["tangle the lemur, sonic the hedgehog \\(series\\)"],
    },
    "sable_able": {
        "character": ["sable_able"],
        "trigger": ["sable able, animal crossing"],
    },
    "cream_heart_(mlp)": {
        "character": ["cream_heart_(mlp)"],
        "trigger": ["cream heart \\(mlp\\), my little pony"],
    },
    "benjamin_clawhauser": {
        "character": ["benjamin_clawhauser"],
        "trigger": ["benjamin clawhauser, disney"],
    },
    "lilo_pelekai": {
        "character": ["lilo_pelekai"],
        "trigger": ["lilo pelekai, disney"],
    },
    "angel_dust": {
        "character": ["angel_dust"],
        "trigger": ["angel dust, hazbin hotel"],
    },
    "pearl_(splatoon)": {
        "character": ["pearl_(splatoon)"],
        "trigger": ["pearl \\(splatoon\\), splatoon"],
    },
    "santa_claus": {
        "character": ["santa_claus"],
        "trigger": ["santa claus, christmas"],
    },
    "master_splinter": {
        "character": ["master_splinter"],
        "trigger": ["master splinter, teenage mutant ninja turtles"],
    },
    "mane_six_(mlp)": {
        "character": ["mane_six_(mlp)"],
        "trigger": ["mane six \\(mlp\\), my little pony"],
    },
    "pirate_tawna": {
        "character": ["pirate_tawna"],
        "trigger": ["pirate tawna, crash bandicoot \\(series\\)"],
    },
    "sasha_(animal_crossing)": {
        "character": ["sasha_(animal_crossing)"],
        "trigger": ["sasha \\(animal crossing\\), animal crossing"],
    },
    "nanachi": {"character": ["nanachi"], "trigger": ["nanachi, made in abyss"]},
    "coco_pommel_(mlp)": {
        "character": ["coco_pommel_(mlp)"],
        "trigger": ["coco pommel \\(mlp\\), hasbro"],
    },
    "claire_(the_summoning)": {
        "character": ["claire_(the_summoning)"],
        "trigger": ["claire \\(the summoning\\), cartoon hangover"],
    },
    "gazelle_(zootopia)": {
        "character": ["gazelle_(zootopia)"],
        "trigger": ["gazelle \\(zootopia\\), disney"],
    },
    "fidget_(elysian_tail)": {
        "character": ["fidget_(elysian_tail)"],
        "trigger": ["fidget \\(elysian tail\\), dust: an elysian tail"],
    },
    "angel_(lilo_and_stitch)": {
        "character": ["angel_(lilo_and_stitch)"],
        "trigger": ["angel \\(lilo and stitch\\), lilo and stitch"],
    },
    "diamond_tiara_(mlp)": {
        "character": ["diamond_tiara_(mlp)"],
        "trigger": ["diamond tiara \\(mlp\\), my little pony"],
    },
    "natani": {"character": ["natani"], "trigger": ["natani, twokinds"]},
    "wendy_o._koopa": {
        "character": ["wendy_o._koopa"],
        "trigger": ["wendy o. koopa, mario bros"],
    },
    "soraka": {"character": ["soraka"], "trigger": ["soraka, riot games"]},
    "keith_keiser": {
        "character": ["keith_keiser"],
        "trigger": ["keith keiser, twokinds"],
    },
    "vanny_(fnaf)": {
        "character": ["vanny_(fnaf)"],
        "trigger": ["vanny \\(fnaf\\), scottgames"],
    },
    "kazooie": {"character": ["kazooie"], "trigger": ["kazooie, rareware"]},
    "gnar_(lol)": {
        "character": ["gnar_(lol)"],
        "trigger": ["gnar \\(lol\\), riot games"],
    },
    "wolf_(lol)": {
        "character": ["wolf_(lol)"],
        "trigger": ["wolf \\(lol\\), riot games"],
    },
    "switch_dog": {
        "character": ["switch_dog"],
        "trigger": ["switch dog, nintendo switch"],
    },
    "kathrin_vaughan": {
        "character": ["kathrin_vaughan"],
        "trigger": ["kathrin vaughan, twokinds"],
    },
    "carrot_(one_piece)": {
        "character": ["carrot_(one_piece)"],
        "trigger": ["carrot \\(one piece\\), one piece"],
    },
    "luigi": {"character": ["luigi"], "trigger": ["luigi, mario bros"]},
    "ran_yakumo": {"character": ["ran_yakumo"], "trigger": ["ran yakumo, touhou"]},
    "jenna_(balto)": {
        "character": ["jenna_(balto)"],
        "trigger": ["jenna \\(balto\\), universal studios"],
    },
    "mokdai": {"character": ["mokdai"], "trigger": ["mokdai, lifewonders"]},
    "ash_ketchum": {"character": ["ash_ketchum"], "trigger": ["ash ketchum, pokemon"]},
    "the_knight_(hollow_knight)": {
        "character": ["the_knight_(hollow_knight)"],
        "trigger": ["the knight \\(hollow knight\\), team cherry"],
    },
    "callie_(splatoon)": {
        "character": ["callie_(splatoon)"],
        "trigger": ["callie \\(splatoon\\), splatoon"],
    },
    "ganglie_(tas)": {
        "character": ["ganglie_(tas)"],
        "trigger": ["ganglie \\(tas\\), lifewonders"],
    },
    "nicole_the_lynx": {
        "character": ["nicole_the_lynx"],
        "trigger": ["nicole the lynx, sonic the hedgehog \\(series\\)"],
    },
    "vex_(lol)": {"character": ["vex_(lol)"], "trigger": ["vex \\(lol\\), riot games"]},
    "shota_feline_(marimo)": {
        "character": ["shota_feline_(marimo)"],
        "trigger": ["shota feline \\(marimo\\), hunter x hunter"],
    },
    "izzy_moonbow_(mlp)": {
        "character": ["izzy_moonbow_(mlp)"],
        "trigger": ["izzy moonbow \\(mlp\\), my little pony"],
    },
    "lammy_lamb": {
        "character": ["lammy_lamb"],
        "trigger": ["lammy lamb, um jammer lammy"],
    },
    "taokaka": {"character": ["taokaka"], "trigger": ["taokaka, arc system works"]},
    "cosplay_pikachu_(character)": {
        "character": ["cosplay_pikachu_(character)"],
        "trigger": ["cosplay pikachu \\(character\\), pokemon"],
    },
    "toy_freddy_(fnaf)": {
        "character": ["toy_freddy_(fnaf)"],
        "trigger": ["toy freddy \\(fnaf\\), five nights at freddy's 2"],
    },
    "rebecca_cunningham": {
        "character": ["rebecca_cunningham"],
        "trigger": ["rebecca cunningham, disney"],
    },
    "flurry_heart_(mlp)": {
        "character": ["flurry_heart_(mlp)"],
        "trigger": ["flurry heart \\(mlp\\), my little pony"],
    },
    "little_red_riding_hood": {
        "character": ["little_red_riding_hood"],
        "trigger": ["little red riding hood, fairy tales"],
    },
    "button_mash_(mlp)": {
        "character": ["button_mash_(mlp)"],
        "trigger": ["button mash \\(mlp\\), my little pony"],
    },
    "li_li_stormstout": {
        "character": ["li_li_stormstout"],
        "trigger": ["li li stormstout, warcraft"],
    },
    "porsha_crystal": {
        "character": ["porsha_crystal"],
        "trigger": ["porsha crystal, illumination entertainment"],
    },
    "dawn_bellwether": {
        "character": ["dawn_bellwether"],
        "trigger": ["dawn bellwether, disney"],
    },
    "max_goof": {"character": ["max_goof"], "trigger": ["max goof, disney"]},
    "brian_griffin": {
        "character": ["brian_griffin"],
        "trigger": ["brian griffin, family guy"],
    },
    "angel_(mlp)": {
        "character": ["angel_(mlp)"],
        "trigger": ["angel \\(mlp\\), my little pony"],
    },
    "torahiko_(morenatsu)": {
        "character": ["torahiko_(morenatsu)"],
        "trigger": ["torahiko \\(morenatsu\\), morenatsu"],
    },
    "penny_fitzgerald": {
        "character": ["penny_fitzgerald"],
        "trigger": ["penny fitzgerald, cartoon network"],
    },
    "ori_(ori)": {
        "character": ["ori_(ori)"],
        "trigger": ["ori \\(ori\\), ori \\(series\\)"],
    },
    "mettaton": {
        "character": ["mettaton"],
        "trigger": ["mettaton, undertale \\(series\\)"],
    },
    "mickey_mouse": {
        "character": ["mickey_mouse"],
        "trigger": ["mickey mouse, disney"],
    },
    "bowser_jr.": {"character": ["bowser_jr."], "trigger": ["bowser jr., mario bros"]},
    "sonic_the_werehog": {
        "character": ["sonic_the_werehog"],
        "trigger": ["sonic the werehog, sonic the hedgehog \\(series\\)"],
    },
    "venom_(marvel)": {
        "character": ["venom_(marvel)"],
        "trigger": ["venom \\(marvel\\), marvel"],
    },
    "silver_spoon_(mlp)": {
        "character": ["silver_spoon_(mlp)"],
        "trigger": ["silver spoon \\(mlp\\), my little pony"],
    },
    "anais_watterson": {
        "character": ["anais_watterson"],
        "trigger": ["anais watterson, the amazing world of gumball"],
    },
    "gallus_(mlp)": {
        "character": ["gallus_(mlp)"],
        "trigger": ["gallus \\(mlp\\), my little pony"],
    },
    "veigar": {"character": ["veigar"], "trigger": ["veigar, riot games"]},
    "minnie_mouse": {
        "character": ["minnie_mouse"],
        "trigger": ["minnie mouse, disney"],
    },
    "sunburst_(mlp)": {
        "character": ["sunburst_(mlp)"],
        "trigger": ["sunburst \\(mlp\\), my little pony"],
    },
    "gregg_lee": {
        "character": ["gregg_lee"],
        "trigger": ["gregg lee, night in the woods"],
    },
    "anonymous": {"character": ["anonymous"], "trigger": ["anonymous, nintendo"]},
    "donkey_kong_(character)": {
        "character": ["donkey_kong_(character)"],
        "trigger": ["donkey kong \\(character\\), donkey kong \\(series\\)"],
    },
    "babs_bunny": {
        "character": ["babs_bunny"],
        "trigger": ["babs bunny, warner brothers"],
    },
    "marceline_abadeer": {
        "character": ["marceline_abadeer"],
        "trigger": ["marceline abadeer, cartoon network"],
    },
    "nasus_(lol)": {
        "character": ["nasus_(lol)"],
        "trigger": ["nasus \\(lol\\), riot games"],
    },
    "gyro_feather": {
        "character": ["gyro_feather"],
        "trigger": ["gyro feather, mythology"],
    },
    "lord_dominator": {
        "character": ["lord_dominator"],
        "trigger": ["lord dominator, wander over yonder"],
    },
    "momiji_inubashiri": {
        "character": ["momiji_inubashiri"],
        "trigger": ["momiji inubashiri, touhou"],
    },
    "revali": {"character": ["revali"], "trigger": ["revali, breath of the wild"]},
    "monster_kid": {
        "character": ["monster_kid"],
        "trigger": ["monster kid, undertale \\(series\\)"],
    },
    "mrs._cake_(mlp)": {
        "character": ["mrs._cake_(mlp)"],
        "trigger": ["mrs. cake \\(mlp\\), my little pony"],
    },
    "cherry_(animal_crossing)": {
        "character": ["cherry_(animal_crossing)"],
        "trigger": ["cherry \\(animal crossing\\), animal crossing"],
    },
    "scar_(the_lion_king)": {
        "character": ["scar_(the_lion_king)"],
        "trigger": ["scar \\(the lion king\\), disney"],
    },
    "warwick_(lol)": {
        "character": ["warwick_(lol)"],
        "trigger": ["warwick \\(lol\\), riot games"],
    },
    "finn_the_human": {
        "character": ["finn_the_human"],
        "trigger": ["finn the human, cartoon network"],
    },
    "shibeta": {"character": ["shibeta"], "trigger": ["shibeta, christmas"]},
    "wolf_(petruz)": {
        "character": ["wolf_(petruz)"],
        "trigger": ["wolf \\(petruz\\), petruz \\(copyright\\)"],
    },
    "slippy_toad": {"character": ["slippy_toad"], "trigger": ["slippy toad, star fox"]},
    "sash_lilac": {
        "character": ["sash_lilac"],
        "trigger": ["sash lilac, freedom planet"],
    },
    "artica_sparkle": {
        "character": ["artica_sparkle"],
        "trigger": ["artica sparkle, mythology"],
    },
    "juuichi_mikazuki": {
        "character": ["juuichi_mikazuki"],
        "trigger": ["juuichi mikazuki, morenatsu"],
    },
    "rika_nonaka": {"character": ["rika_nonaka"], "trigger": ["rika nonaka, digimon"]},
    "zig_zag": {"character": ["zig_zag"], "trigger": ["zig zag, sabrina online"]},
    "yuumi_(lol)": {
        "character": ["yuumi_(lol)"],
        "trigger": ["yuumi \\(lol\\), riot games"],
    },
    "big_the_cat": {
        "character": ["big_the_cat"],
        "trigger": ["big the cat, sonic the hedgehog \\(series\\)"],
    },
    "cu_sith_(tas)": {
        "character": ["cu_sith_(tas)"],
        "trigger": ["cu sith \\(tas\\), lifewonders"],
    },
    "mina_mongoose": {
        "character": ["mina_mongoose"],
        "trigger": ["mina mongoose, sonic the hedgehog \\(series\\)"],
    },
    "neco-arc": {"character": ["neco-arc"], "trigger": ["neco-arc, tsukihime"]},
    "max_(sam_and_max)": {
        "character": ["max_(sam_and_max)"],
        "trigger": ["max \\(sam and max\\), sam and max"],
    },
    "ridley": {"character": ["ridley"], "trigger": ["ridley, nintendo"]},
    "holo_(spice_and_wolf)": {
        "character": ["holo_(spice_and_wolf)"],
        "trigger": ["holo \\(spice and wolf\\), spice and wolf"],
    },
    "ilulu": {
        "character": ["ilulu"],
        "trigger": ["ilulu, miss kobayashi's dragon maid"],
    },
    "princess_bubblegum": {
        "character": ["princess_bubblegum"],
        "trigger": ["princess bubblegum, cartoon network"],
    },
    "sunny_starscout_(mlp)": {
        "character": ["sunny_starscout_(mlp)"],
        "trigger": ["sunny starscout \\(mlp\\), my little pony"],
    },
    "rengar_(lol)": {
        "character": ["rengar_(lol)"],
        "trigger": ["rengar \\(lol\\), riot games"],
    },
    "balto": {"character": ["balto"], "trigger": ["balto, universal studios"]},
    "jon_talbain": {
        "character": ["jon_talbain"],
        "trigger": ["jon talbain, darkstalkers"],
    },
    "dr._eggman": {
        "character": ["dr._eggman"],
        "trigger": ["dr. eggman, sonic the hedgehog \\(series\\)"],
    },
    "momo_(google)": {
        "character": ["momo_(google)"],
        "trigger": ["momo \\(google\\), google"],
    },
    "meowscles": {"character": ["meowscles"], "trigger": ["meowscles, fortnite"]},
    "may_(pokemon)": {
        "character": ["may_(pokemon)"],
        "trigger": ["may \\(pokemon\\), pokemon"],
    },
    "kiara_(the_lion_king)": {
        "character": ["kiara_(the_lion_king)"],
        "trigger": ["kiara \\(the lion king\\), disney"],
    },
    "volos_(tas)": {
        "character": ["volos_(tas)"],
        "trigger": ["volos \\(tas\\), lifewonders"],
    },
    "pipp_petals_(mlp)": {
        "character": ["pipp_petals_(mlp)"],
        "trigger": ["pipp petals \\(mlp\\), my little pony"],
    },
    "amicus_(adastra)": {
        "character": ["amicus_(adastra)"],
        "trigger": ["amicus \\(adastra\\), adastra \\(series\\)"],
    },
    "king_k._rool": {
        "character": ["king_k._rool"],
        "trigger": ["king k. rool, donkey kong \\(series\\)"],
    },
    "fiona_fox": {
        "character": ["fiona_fox"],
        "trigger": ["fiona fox, sonic the hedgehog \\(series\\)"],
    },
    "limestone_pie_(mlp)": {
        "character": ["limestone_pie_(mlp)"],
        "trigger": ["limestone pie \\(mlp\\), my little pony"],
    },
    "marshal_(animal_crossing)": {
        "character": ["marshal_(animal_crossing)"],
        "trigger": ["marshal \\(animal crossing\\), animal crossing"],
    },
    "catty_(undertale)": {
        "character": ["catty_(undertale)"],
        "trigger": ["catty \\(undertale\\), undertale \\(series\\)"],
    },
    "tony_tony_chopper": {
        "character": ["tony_tony_chopper"],
        "trigger": ["tony tony chopper, one piece"],
    },
    "scooby-doo": {
        "character": ["scooby-doo"],
        "trigger": ["scooby-doo, scooby-doo \\(series\\)"],
    },
    "gyobu_(tas)": {
        "character": ["gyobu_(tas)"],
        "trigger": ["gyobu \\(tas\\), lifewonders"],
    },
    "jake_long": {"character": ["jake_long"], "trigger": ["jake long, disney"]},
    "lulu_(lol)": {
        "character": ["lulu_(lol)"],
        "trigger": ["lulu \\(lol\\), riot games"],
    },
    "kurama": {"character": ["kurama"], "trigger": ["kurama, naruto"]},
    "brok_(brok_the_investigator)": {
        "character": ["brok_(brok_the_investigator)"],
        "trigger": ["brok \\(brok the investigator\\), brok the investigator"],
    },
    "kyappy": {"character": ["kyappy"], "trigger": ["kyappy, christmas"]},
    "fizz_(lol)": {
        "character": ["fizz_(lol)"],
        "trigger": ["fizz \\(lol\\), riot games"],
    },
    "rudolph_the_red-nosed_reindeer": {
        "character": ["rudolph_the_red-nosed_reindeer"],
        "trigger": ["rudolph the red-nosed reindeer, christmas"],
    },
    "golden_freddy_(fnaf)": {
        "character": ["golden_freddy_(fnaf)"],
        "trigger": ["golden freddy \\(fnaf\\), scottgames"],
    },
    "babs_seed_(mlp)": {
        "character": ["babs_seed_(mlp)"],
        "trigger": ["babs seed \\(mlp\\), my little pony"],
    },
    "sisu_(ratld)": {
        "character": ["sisu_(ratld)"],
        "trigger": ["sisu \\(ratld\\), raya and the last dragon"],
    },
    "inkling_girl": {
        "character": ["inkling_girl"],
        "trigger": ["inkling girl, splatoon"],
    },
    "misty_(pokemon)": {
        "character": ["misty_(pokemon)"],
        "trigger": ["misty \\(pokemon\\), pokemon"],
    },
    "nika_sharkeh": {
        "character": ["nika_sharkeh"],
        "trigger": ["nika sharkeh, source filmmaker"],
    },
    "venus_spring": {
        "character": ["venus_spring"],
        "trigger": ["venus spring, my little pony"],
    },
    "mrs._katswell": {
        "character": ["mrs._katswell"],
        "trigger": ["mrs. katswell, t.u.f.f. puppy"],
    },
    "tadatomo_(tas)": {
        "character": ["tadatomo_(tas)"],
        "trigger": ["tadatomo \\(tas\\), lifewonders"],
    },
    "thunderlane_(mlp)": {
        "character": ["thunderlane_(mlp)"],
        "trigger": ["thunderlane \\(mlp\\), my little pony"],
    },
    "c.j._(animal_crossing)": {
        "character": ["c.j._(animal_crossing)"],
        "trigger": ["c.j. \\(animal crossing\\), animal crossing"],
    },
    "lin_(changed)": {
        "character": ["lin_(changed)"],
        "trigger": ["lin \\(changed\\), changed \\(video game\\)"],
    },
    "white_canine_(marimo)": {
        "character": ["white_canine_(marimo)"],
        "trigger": ["white canine \\(marimo\\), hunter x hunter"],
    },
    "seth_(tas)": {
        "character": ["seth_(tas)"],
        "trigger": ["seth \\(tas\\), lifewonders"],
    },
    "dawn_(pokemon)": {
        "character": ["dawn_(pokemon)"],
        "trigger": ["dawn \\(pokemon\\), pokemon"],
    },
    "spongebob_squarepants_(character)": {
        "character": ["spongebob_squarepants_(character)"],
        "trigger": ["spongebob squarepants \\(character\\), nickelodeon"],
    },
    "baloo": {"character": ["baloo"], "trigger": ["baloo, the jungle book"]},
    "aeris_(vg_cats)": {
        "character": ["aeris_(vg_cats)"],
        "trigger": ["aeris \\(vg cats\\), vg cats"],
    },
    "vortex_(helluva_boss)": {
        "character": ["vortex_(helluva_boss)"],
        "trigger": ["vortex \\(helluva boss\\), helluva boss"],
    },
    "reuben_(lilo_and_stitch)": {
        "character": ["reuben_(lilo_and_stitch)"],
        "trigger": ["reuben \\(lilo and stitch\\), disney"],
    },
    "garfield_the_cat": {
        "character": ["garfield_the_cat"],
        "trigger": ["garfield the cat, garfield \\(series\\)"],
    },
    "princess_daisy": {
        "character": ["princess_daisy"],
        "trigger": ["princess daisy, mario bros"],
    },
    "conker": {"character": ["conker"], "trigger": ["conker, conker's bad fur day"]},
    "jenny_(bucky_o'hare)": {
        "character": ["jenny_(bucky_o'hare)"],
        "trigger": ["jenny \\(bucky o'hare\\), bucky o'hare \\(series\\)"],
    },
    "daring_do_(mlp)": {
        "character": ["daring_do_(mlp)"],
        "trigger": ["daring do \\(mlp\\), my little pony"],
    },
    "raven_(dc)": {
        "character": ["raven_(dc)"],
        "trigger": ["raven \\(dc\\), dc comics"],
    },
    "bolt_(bolt)": {
        "character": ["bolt_(bolt)"],
        "trigger": ["bolt \\(bolt\\), disney"],
    },
    "richard_watterson": {
        "character": ["richard_watterson"],
        "trigger": ["richard watterson, cartoon network"],
    },
    "ashigara_(tas)": {
        "character": ["ashigara_(tas)"],
        "trigger": ["ashigara \\(tas\\), lifewonders"],
    },
    "renekton": {"character": ["renekton"], "trigger": ["renekton, riot games"]},
    "apogee_(tinygaypirate)": {
        "character": ["apogee_(tinygaypirate)"],
        "trigger": ["apogee \\(tinygaypirate\\), strip meme"],
    },
    "rescued_dragons_(spyro)": {
        "character": ["rescued_dragons_(spyro)"],
        "trigger": ["rescued dragons \\(spyro\\), mythology"],
    },
    "hilda_(pokemon)": {
        "character": ["hilda_(pokemon)"],
        "trigger": ["hilda \\(pokemon\\), pokemon"],
    },
    "trish_(gvh)": {
        "character": ["trish_(gvh)"],
        "trigger": ["trish \\(gvh\\), goodbye volcano high"],
    },
    "trace_legacy": {
        "character": ["trace_legacy"],
        "trigger": ["trace legacy, twokinds"],
    },
    "kuruk_(character)": {
        "character": ["kuruk_(character)"],
        "trigger": ["kuruk \\(character\\), mythology"],
    },
    "sherri_mayim": {
        "character": ["sherri_mayim"],
        "trigger": ["sherri mayim, hasbro"],
    },
    "miyu_(star_fox)": {
        "character": ["miyu_(star_fox)"],
        "trigger": ["miyu \\(star fox\\), star fox"],
    },
    "marine_the_raccoon": {
        "character": ["marine_the_raccoon"],
        "trigger": ["marine the raccoon, sonic the hedgehog \\(series\\)"],
    },
    "marionette_(fnaf)": {
        "character": ["marionette_(fnaf)"],
        "trigger": ["marionette \\(fnaf\\), five nights at freddy's 2"],
    },
    "prince_sidon": {
        "character": ["prince_sidon"],
        "trigger": ["prince sidon, the legend of zelda"],
    },
    "roxanne_(goof_troop)": {
        "character": ["roxanne_(goof_troop)"],
        "trigger": ["roxanne \\(goof troop\\), disney"],
    },
    "big_bad_wolf": {
        "character": ["big_bad_wolf"],
        "trigger": ["big bad wolf, fairy tales"],
    },
    "red_xiii": {"character": ["red_xiii"], "trigger": ["red xiii, final fantasy vii"]},
    "doctor_whooves_(mlp)": {
        "character": ["doctor_whooves_(mlp)"],
        "trigger": ["doctor whooves \\(mlp\\), my little pony"],
    },
    "twitch_(lol)": {
        "character": ["twitch_(lol)"],
        "trigger": ["twitch \\(lol\\), riot games"],
    },
    "licho_(tas)": {
        "character": ["licho_(tas)"],
        "trigger": ["licho \\(tas\\), lifewonders"],
    },
    "buster_bunny": {
        "character": ["buster_bunny"],
        "trigger": ["buster bunny, warner brothers"],
    },
    "kovu_(the_lion_king)": {
        "character": ["kovu_(the_lion_king)"],
        "trigger": ["kovu \\(the lion king\\), disney"],
    },
    "brandy_harrington": {
        "character": ["brandy_harrington"],
        "trigger": ["brandy harrington, disney"],
    },
    "krampus": {"character": ["krampus"], "trigger": ["krampus, christmas"]},
    "dot_warner": {
        "character": ["dot_warner"],
        "trigger": ["dot warner, warner brothers"],
    },
    "ashido_mina": {
        "character": ["ashido_mina"],
        "trigger": ["ashido mina, my hero academia"],
    },
    "nurse_redheart_(mlp)": {
        "character": ["nurse_redheart_(mlp)"],
        "trigger": ["nurse redheart \\(mlp\\), my little pony"],
    },
    "pirate_eagle": {
        "character": ["pirate_eagle"],
        "trigger": ["pirate eagle, mythology"],
    },
    "green_fox_(foxes_in_love)": {
        "character": ["green_fox_(foxes_in_love)"],
        "trigger": ["green fox \\(foxes in love\\), foxes in love"],
    },
    "goofy_(disney)": {
        "character": ["goofy_(disney)"],
        "trigger": ["goofy \\(disney\\), disney"],
    },
    "commander_shepard": {
        "character": ["commander_shepard"],
        "trigger": ["commander shepard, mass effect"],
    },
    "lizard_(petruz)": {
        "character": ["lizard_(petruz)"],
        "trigger": ["lizard \\(petruz\\), petruz \\(copyright\\)"],
    },
    "liara_t'soni": {
        "character": ["liara_t'soni"],
        "trigger": ["liara t'soni, mass effect"],
    },
    "ms._harshwhinny_(mlp)": {
        "character": ["ms._harshwhinny_(mlp)"],
        "trigger": ["ms. harshwhinny \\(mlp\\), my little pony"],
    },
    "alp_(tas)": {
        "character": ["alp_(tas)"],
        "trigger": ["alp \\(tas\\), lifewonders"],
    },
    "foxy_(psychojohn2)": {
        "character": ["foxy_(psychojohn2)"],
        "trigger": ["foxy \\(psychojohn2\\), scottgames"],
    },
    "daybreaker_(mlp)": {
        "character": ["daybreaker_(mlp)"],
        "trigger": ["daybreaker \\(mlp\\), my little pony"],
    },
    "barrel_(live_a_hero)": {
        "character": ["barrel_(live_a_hero)"],
        "trigger": ["barrel \\(live a hero\\), lifewonders"],
    },
    "firondraak": {"character": ["firondraak"], "trigger": ["firondraak, mythology"]},
    "funtime_foxy_(fnaf)": {
        "character": ["funtime_foxy_(fnaf)"],
        "trigger": ["funtime foxy \\(fnaf\\), scottgames"],
    },
    "peppy_hare": {"character": ["peppy_hare"], "trigger": ["peppy hare, star fox"]},
    "blue_fox_(foxes_in_love)": {
        "character": ["blue_fox_(foxes_in_love)"],
        "trigger": ["blue fox \\(foxes in love\\), foxes in love"],
    },
    "lotus_(mlp)": {
        "character": ["lotus_(mlp)"],
        "trigger": ["lotus \\(mlp\\), my little pony"],
    },
    "tweetfur": {"character": ["tweetfur"], "trigger": ["tweetfur, twitter"]},
    "aloe_(mlp)": {
        "character": ["aloe_(mlp)"],
        "trigger": ["aloe \\(mlp\\), my little pony"],
    },
    "nightmare_rarity_(idw)": {
        "character": ["nightmare_rarity_(idw)"],
        "trigger": ["nightmare rarity \\(idw\\), idw publishing"],
    },
    "ganondorf": {
        "character": ["ganondorf"],
        "trigger": ["ganondorf, the legend of zelda"],
    },
    "loona_(aeridiccore)": {
        "character": ["loona_(aeridiccore)"],
        "trigger": ["loona \\(aeridiccore\\), helluva boss"],
    },
    "nazuna_hiwatashi": {
        "character": ["nazuna_hiwatashi"],
        "trigger": ["nazuna hiwatashi, studio trigger"],
    },
    "cupcake_(fnaf)": {
        "character": ["cupcake_(fnaf)"],
        "trigger": ["cupcake \\(fnaf\\), scottgames"],
    },
    "charlie_morningstar": {
        "character": ["charlie_morningstar"],
        "trigger": ["charlie morningstar, hazbin hotel"],
    },
    "fennix_(fortnite)": {
        "character": ["fennix_(fortnite)"],
        "trigger": ["fennix \\(fortnite\\), fortnite"],
    },
    "night_light_(mlp)": {
        "character": ["night_light_(mlp)"],
        "trigger": ["night light \\(mlp\\), my little pony"],
    },
    "tifa_lockhart": {
        "character": ["tifa_lockhart"],
        "trigger": ["tifa lockhart, final fantasy vii"],
    },
    "ahsoka_tano": {
        "character": ["ahsoka_tano"],
        "trigger": ["ahsoka tano, star wars"],
    },
    "alex_marx": {
        "character": ["alex_marx"],
        "trigger": ["alex marx, sony interactive entertainment"],
    },
    "stella_(gvh_beta)": {
        "character": ["stella_(gvh_beta)"],
        "trigger": ["stella \\(gvh beta\\), goodbye volcano high"],
    },
    "pepe_le_pew": {
        "character": ["pepe_le_pew"],
        "trigger": ["pepe le pew, looney tunes"],
    },
    "tony_the_tiger": {
        "character": ["tony_the_tiger"],
        "trigger": ["tony the tiger, kellogg's"],
    },
    "kaa_(jungle_book)": {
        "character": ["kaa_(jungle_book)"],
        "trigger": ["kaa \\(jungle book\\), the jungle book"],
    },
    "vivian_(mario)": {
        "character": ["vivian_(mario)"],
        "trigger": ["vivian \\(mario\\), mario bros"],
    },
    "chance_furlong": {
        "character": ["chance_furlong"],
        "trigger": ["chance furlong, swat kats"],
    },
    "zipp_storm_(mlp)": {
        "character": ["zipp_storm_(mlp)"],
        "trigger": ["zipp storm \\(mlp\\), my little pony"],
    },
    "catnap_(poppy_playtime)": {
        "character": ["catnap_(poppy_playtime)"],
        "trigger": ["catnap \\(poppy playtime\\), poppy playtime"],
    },
    "busty_bird": {"character": ["busty_bird"], "trigger": ["busty bird, halloween"]},
    "mad_rat_(mad_rat_dead)": {
        "character": ["mad_rat_(mad_rat_dead)"],
        "trigger": ["mad rat \\(mad rat dead\\), nippon ichi software"],
    },
    "baphomet_(deity)": {
        "character": ["baphomet_(deity)"],
        "trigger": ["baphomet \\(deity\\), mythology"],
    },
    "fleur_de_lis_(mlp)": {
        "character": ["fleur_de_lis_(mlp)"],
        "trigger": ["fleur de lis \\(mlp\\), my little pony"],
    },
    "jet_the_hawk": {
        "character": ["jet_the_hawk"],
        "trigger": ["jet the hawk, sonic riders"],
    },
    "raven_team_leader": {
        "character": ["raven_team_leader"],
        "trigger": ["raven team leader, fortnite"],
    },
    "anonymous_character": {
        "character": ["anonymous_character"],
        "trigger": ["anonymous character, mythology"],
    },
    "rosalina_(mario)": {
        "character": ["rosalina_(mario)"],
        "trigger": ["rosalina \\(mario\\), mario bros"],
    },
    "donald_duck": {"character": ["donald_duck"], "trigger": ["donald duck, disney"]},
    "ashley_graham_(resident_evil)": {
        "character": ["ashley_graham_(resident_evil)"],
        "trigger": ["ashley graham \\(resident evil\\), resident evil"],
    },
    "unikitty": {"character": ["unikitty"], "trigger": ["unikitty, the lego movie"]},
    "berry_punch_(mlp)": {
        "character": ["berry_punch_(mlp)"],
        "trigger": ["berry punch \\(mlp\\), my little pony"],
    },
    "vector_the_crocodile": {
        "character": ["vector_the_crocodile"],
        "trigger": ["vector the crocodile, sonic the hedgehog \\(series\\)"],
    },
    "sponty": {"character": ["sponty"], "trigger": ["sponty, mythology"]},
    "king_dedede": {
        "character": ["king_dedede"],
        "trigger": ["king dedede, kirby \\(series\\)"],
    },
    "teemo_the_yiffer": {
        "character": ["teemo_the_yiffer"],
        "trigger": ["teemo the yiffer, riot games"],
    },
    "april_o'neil": {
        "character": ["april_o'neil"],
        "trigger": ["april o'neil, teenage mutant ninja turtles"],
    },
    "sarabi_(the_lion_king)": {
        "character": ["sarabi_(the_lion_king)"],
        "trigger": ["sarabi \\(the lion king\\), disney"],
    },
    "polly_esther": {
        "character": ["polly_esther"],
        "trigger": ["polly esther, samurai pizza cats"],
    },
    "belladonna_(trials_of_mana)": {
        "character": ["belladonna_(trials_of_mana)"],
        "trigger": ["belladonna \\(trials of mana\\), trials of mana"],
    },
    "samuel_dog": {"character": ["samuel_dog"], "trigger": ["samuel dog, sam and max"]},
    "mangle_(psychojohn2)": {
        "character": ["mangle_(psychojohn2)"],
        "trigger": ["mangle \\(psychojohn2\\), five nights at freddy's 2"],
    },
    "azzilan": {"character": ["azzilan"], "trigger": ["azzilan, pokemon"]},
    "garrus_vakarian": {
        "character": ["garrus_vakarian"],
        "trigger": ["garrus vakarian, mass effect"],
    },
    "haydee": {"character": ["haydee"], "trigger": ["haydee, haydee \\(game\\)"]},
    "quetzalcoatl_(dragon_maid)": {
        "character": ["quetzalcoatl_(dragon_maid)"],
        "trigger": ["quetzalcoatl \\(dragon maid\\), miss kobayashi's dragon maid"],
    },
    "morrigan_aensland": {
        "character": ["morrigan_aensland"],
        "trigger": ["morrigan aensland, darkstalkers"],
    },
    "moondancer_(mlp)": {
        "character": ["moondancer_(mlp)"],
        "trigger": ["moondancer \\(mlp\\), my little pony"],
    },
    "autumn_blaze_(mlp)": {
        "character": ["autumn_blaze_(mlp)"],
        "trigger": ["autumn blaze \\(mlp\\), my little pony"],
    },
    "kouya_(morenatsu)": {
        "character": ["kouya_(morenatsu)"],
        "trigger": ["kouya \\(morenatsu\\), morenatsu"],
    },
    "skyrim_werewolf": {
        "character": ["skyrim_werewolf"],
        "trigger": ["skyrim werewolf, mythology"],
    },
    "tali'zorah": {"character": ["tali'zorah"], "trigger": ["tali'zorah, mass effect"]},
    "kion_(the_lion_guard)": {
        "character": ["kion_(the_lion_guard)"],
        "trigger": ["kion \\(the lion guard\\), disney"],
    },
    "zorori": {"character": ["zorori"], "trigger": ["zorori, kaiketsu zorori"]},
    "honey_the_cat": {
        "character": ["honey_the_cat"],
        "trigger": ["honey the cat, sonic the hedgehog \\(series\\)"],
    },
    "ryekie_(live_a_hero)": {
        "character": ["ryekie_(live_a_hero)"],
        "trigger": ["ryekie \\(live a hero\\), lifewonders"],
    },
    "goon_(goonie_san)": {
        "character": ["goon_(goonie_san)"],
        "trigger": ["goon \\(goonie san\\), nintendo"],
    },
    "clawroline": {
        "character": ["clawroline"],
        "trigger": ["clawroline, kirby \\(series\\)"],
    },
    "jax_(tadc)": {
        "character": ["jax_(tadc)"],
        "trigger": ["jax \\(tadc\\), the amazing digital circus"],
    },
    "naomi_(gvh)": {
        "character": ["naomi_(gvh)"],
        "trigger": ["naomi \\(gvh\\), goodbye volcano high"],
    },
    "crewmate_(among_us)": {
        "character": ["crewmate_(among_us)"],
        "trigger": ["crewmate \\(among us\\), among us"],
    },
    "carol_tea": {"character": ["carol_tea"], "trigger": ["carol tea, freedom planet"]},
    "yuki_(evov1)": {
        "character": ["yuki_(evov1)"],
        "trigger": ["yuki \\(evov1\\), pokemon"],
    },
    "rocko_rama": {
        "character": ["rocko_rama"],
        "trigger": ["rocko rama, rocko's modern life"],
    },
    "nifram_logan": {
        "character": ["nifram_logan"],
        "trigger": ["nifram logan, mythology"],
    },
    "hitch_trailblazer_(mlp)": {
        "character": ["hitch_trailblazer_(mlp)"],
        "trigger": ["hitch trailblazer \\(mlp\\), my little pony"],
    },
    "broodal": {"character": ["broodal"], "trigger": ["broodal, super mario odyssey"]},
    "bathym_(tas)": {
        "character": ["bathym_(tas)"],
        "trigger": ["bathym \\(tas\\), lifewonders"],
    },
    "mufasa": {"character": ["mufasa"], "trigger": ["mufasa, disney"]},
    "blue_(jurassic_world)": {
        "character": ["blue_(jurassic_world)"],
        "trigger": ["blue \\(jurassic world\\), universal studios"],
    },
    "nadia_fortune": {
        "character": ["nadia_fortune"],
        "trigger": ["nadia fortune, skullgirls"],
    },
    "tsunoda_(aggretsuko)": {
        "character": ["tsunoda_(aggretsuko)"],
        "trigger": ["tsunoda \\(aggretsuko\\), sanrio"],
    },
    "volibear": {"character": ["volibear"], "trigger": ["volibear, riot games"]},
    "mettaton_ex": {
        "character": ["mettaton_ex"],
        "trigger": ["mettaton ex, undertale \\(series\\)"],
    },
    "mayor_mare_(mlp)": {
        "character": ["mayor_mare_(mlp)"],
        "trigger": ["mayor mare \\(mlp\\), my little pony"],
    },
    "a-chan": {"character": ["a-chan"], "trigger": ["a-chan, april fools' day"]},
    "firecat": {"character": ["firecat"], "trigger": ["firecat, nintendo"]},
    "hatsune_miku": {
        "character": ["hatsune_miku"],
        "trigger": ["hatsune miku, vocaloid"],
    },
    "velvet_reindeer_(tfh)": {
        "character": ["velvet_reindeer_(tfh)"],
        "trigger": ["velvet reindeer \\(tfh\\), them's fightin' herds"],
    },
    "hung_(arknights)": {
        "character": ["hung_(arknights)"],
        "trigger": ["hung \\(arknights\\), studio montagne"],
    },
    "queen_bee-lzebub_(helluva_boss)": {
        "character": ["queen_bee-lzebub_(helluva_boss)"],
        "trigger": ["queen bee-lzebub \\(helluva boss\\), helluva boss"],
    },
    "lolbit_(fnaf)": {
        "character": ["lolbit_(fnaf)"],
        "trigger": ["lolbit \\(fnaf\\), scottgames"],
    },
    "kanna_kamui": {
        "character": ["kanna_kamui"],
        "trigger": ["kanna kamui, miss kobayashi's dragon maid"],
    },
    "fenavi_montaro": {
        "character": ["fenavi_montaro"],
        "trigger": ["fenavi montaro, tale of tails"],
    },
    "angus_delaney": {
        "character": ["angus_delaney"],
        "trigger": ["angus delaney, night in the woods"],
    },
    "daffy_duck": {
        "character": ["daffy_duck"],
        "trigger": ["daffy duck, looney tunes"],
    },
    "nightshade_(kadath)": {
        "character": ["nightshade_(kadath)"],
        "trigger": ["nightshade \\(kadath\\), patreon"],
    },
    "darwin_watterson": {
        "character": ["darwin_watterson"],
        "trigger": ["darwin watterson, cartoon network"],
    },
    "princess_ruto": {
        "character": ["princess_ruto"],
        "trigger": ["princess ruto, the legend of zelda"],
    },
    "emelie_(cyancapsule)": {
        "character": ["emelie_(cyancapsule)"],
        "trigger": ["emelie \\(cyancapsule\\), my pig princess"],
    },
    "marci_hetson": {
        "character": ["marci_hetson"],
        "trigger": ["marci hetson, if hell had a taste"],
    },
    "bambi": {"character": ["bambi"], "trigger": ["bambi, disney"]},
    "alvin_seville": {
        "character": ["alvin_seville"],
        "trigger": ["alvin seville, alvin and the chipmunks"],
    },
    "raine_silverlock": {
        "character": ["raine_silverlock"],
        "trigger": ["raine silverlock, twokinds"],
    },
    "xero_(captainscales)": {
        "character": ["xero_(captainscales)"],
        "trigger": ["xero \\(captainscales\\), mythology"],
    },
    "snails_(mlp)": {
        "character": ["snails_(mlp)"],
        "trigger": ["snails \\(mlp\\), my little pony"],
    },
    "shenzi_(the_lion_king)": {
        "character": ["shenzi_(the_lion_king)"],
        "trigger": ["shenzi \\(the lion king\\), disney"],
    },
    "angel_(lady_and_the_tramp)": {
        "character": ["angel_(lady_and_the_tramp)"],
        "trigger": ["angel \\(lady and the tramp\\), disney"],
    },
    "casimira_(orannis0)": {
        "character": ["casimira_(orannis0)"],
        "trigger": ["casimira \\(orannis0\\), halloween"],
    },
    "meow_(space_dandy)": {
        "character": ["meow_(space_dandy)"],
        "trigger": ["meow \\(space dandy\\), space dandy"],
    },
    "thomas_cat": {
        "character": ["thomas_cat"],
        "trigger": ["thomas cat, metro-goldwyn-mayer"],
    },
    "bob_(animal_crossing)": {
        "character": ["bob_(animal_crossing)"],
        "trigger": ["bob \\(animal crossing\\), animal crossing"],
    },
    "kyaru_(princess_connect!)": {
        "character": ["kyaru_(princess_connect!)"],
        "trigger": ["kyaru \\(princess connect!\\), cygames"],
    },
    "champa": {"character": ["champa"], "trigger": ["champa, dragon ball"]},
    "shima_luan": {
        "character": ["shima_luan"],
        "trigger": ["shima luan, super planet dolan"],
    },
    "beast_boy": {"character": ["beast_boy"], "trigger": ["beast boy, dc comics"]},
    "mora_linda": {"character": ["mora_linda"], "trigger": ["mora linda, las lindas"]},
    "leo_(vg_cats)": {
        "character": ["leo_(vg_cats)"],
        "trigger": ["leo \\(vg cats\\), vg cats"],
    },
    "captain_amelia": {
        "character": ["captain_amelia"],
        "trigger": ["captain amelia, disney"],
    },
    "circus_baby_(fnaf)": {
        "character": ["circus_baby_(fnaf)"],
        "trigger": ["circus baby \\(fnaf\\), scottgames"],
    },
    "jerry_mouse": {
        "character": ["jerry_mouse"],
        "trigger": ["jerry mouse, metro-goldwyn-mayer"],
    },
    "rumble_(mlp)": {
        "character": ["rumble_(mlp)"],
        "trigger": ["rumble \\(mlp\\), my little pony"],
    },
    "polar_patroller": {
        "character": ["polar_patroller"],
        "trigger": ["polar patroller, fortnite"],
    },
    "hekapoo": {"character": ["hekapoo"], "trigger": ["hekapoo, disney"]},
    "teba_(tloz)": {
        "character": ["teba_(tloz)"],
        "trigger": ["teba \\(tloz\\), the legend of zelda"],
    },
    "rauru_(tears_of_the_kingdom)": {
        "character": ["rauru_(tears_of_the_kingdom)"],
        "trigger": ["rauru \\(tears of the kingdom\\), the legend of zelda"],
    },
    "steele_(balto)": {
        "character": ["steele_(balto)"],
        "trigger": ["steele \\(balto\\), universal studios"],
    },
    "milla_basset": {
        "character": ["milla_basset"],
        "trigger": ["milla basset, freedom planet"],
    },
    "tsathoggua_(tas)": {
        "character": ["tsathoggua_(tas)"],
        "trigger": ["tsathoggua \\(tas\\), lifewonders"],
    },
    "rover_(animal_crossing)": {
        "character": ["rover_(animal_crossing)"],
        "trigger": ["rover \\(animal crossing\\), animal crossing"],
    },
    "brooklyn_(gargoyles)": {
        "character": ["brooklyn_(gargoyles)"],
        "trigger": ["brooklyn \\(gargoyles\\), disney"],
    },
    "hex_maniac": {"character": ["hex_maniac"], "trigger": ["hex maniac, pokemon"]},
    "hariet_(mario)": {
        "character": ["hariet_(mario)"],
        "trigger": ["hariet \\(mario\\), super mario odyssey"],
    },
    "mr._snake_(the_bad_guys)": {
        "character": ["mr._snake_(the_bad_guys)"],
        "trigger": ["mr. snake \\(the bad guys\\), the bad guys"],
    },
    "shiron": {"character": ["shiron"], "trigger": ["shiron, legendz"]},
    "pyro_(team_fortress_2)": {
        "character": ["pyro_(team_fortress_2)"],
        "trigger": ["pyro \\(team fortress 2\\), valve"],
    },
    "diddy_kong": {
        "character": ["diddy_kong"],
        "trigger": ["diddy kong, donkey kong \\(series\\)"],
    },
    "lilith_calah": {
        "character": ["lilith_calah"],
        "trigger": ["lilith calah, dreamkeepers"],
    },
    "kalnareff_(character)": {
        "character": ["kalnareff_(character)"],
        "trigger": ["kalnareff \\(character\\), mythology"],
    },
    "adorabat": {"character": ["adorabat"], "trigger": ["adorabat, cartoon network"]},
    "queen_tyr'ahnee": {
        "character": ["queen_tyr'ahnee"],
        "trigger": ["queen tyr'ahnee, duck dodgers"],
    },
    "silverstream_(mlp)": {
        "character": ["silverstream_(mlp)"],
        "trigger": ["silverstream \\(mlp\\), my little pony"],
    },
    "toon_link": {
        "character": ["toon_link"],
        "trigger": ["toon link, the legend of zelda"],
    },
    "dolly_(101_dalmatians)": {
        "character": ["dolly_(101_dalmatians)"],
        "trigger": ["dolly \\(101 dalmatians\\), disney"],
    },
    "berdly": {"character": ["berdly"], "trigger": ["berdly, undertale \\(series\\)"]},
    "cheese_the_chao": {
        "character": ["cheese_the_chao"],
        "trigger": ["cheese the chao, sonic the hedgehog \\(series\\)"],
    },
    "cloud_chaser_(mlp)": {
        "character": ["cloud_chaser_(mlp)"],
        "trigger": ["cloud chaser \\(mlp\\), my little pony"],
    },
    "moushley": {"character": ["moushley"], "trigger": ["moushley, resident evil"]},
    "doom_slayer": {
        "character": ["doom_slayer"],
        "trigger": ["doom slayer, id software"],
    },
    "sandbar_(mlp)": {
        "character": ["sandbar_(mlp)"],
        "trigger": ["sandbar \\(mlp\\), my little pony"],
    },
    "verosika_mayday_(helluva_boss)": {
        "character": ["verosika_mayday_(helluva_boss)"],
        "trigger": ["verosika mayday \\(helluva boss\\), helluva boss"],
    },
    "skye_(paw_patrol)": {
        "character": ["skye_(paw_patrol)"],
        "trigger": ["skye \\(paw patrol\\), paw patrol"],
    },
    "averi_(fiddleafox)": {
        "character": ["averi_(fiddleafox)"],
        "trigger": ["averi \\(fiddleafox\\), 4chan"],
    },
    "dire_(fortnite)": {
        "character": ["dire_(fortnite)"],
        "trigger": ["dire \\(fortnite\\), fortnite"],
    },
    "thorax_(mlp)": {
        "character": ["thorax_(mlp)"],
        "trigger": ["thorax \\(mlp\\), my little pony"],
    },
    "bluey_heeler": {
        "character": ["bluey_heeler"],
        "trigger": ["bluey heeler, bluey \\(series\\)"],
    },
    "beau_(animal_crossing)": {
        "character": ["beau_(animal_crossing)"],
        "trigger": ["beau \\(animal crossing\\), animal crossing"],
    },
    "tiger_dancer_(zootopia)": {
        "character": ["tiger_dancer_(zootopia)"],
        "trigger": ["tiger dancer \\(zootopia\\), disney"],
    },
    "adelia_(changbae)": {
        "character": ["adelia_(changbae)"],
        "trigger": ["adelia \\(changbae\\), mythology"],
    },
    "orisa_(overwatch)": {
        "character": ["orisa_(overwatch)"],
        "trigger": ["orisa \\(overwatch\\), blizzard entertainment"],
    },
    "ruri_tsukiyono": {
        "character": ["ruri_tsukiyono"],
        "trigger": ["ruri tsukiyono, digimon"],
    },
    "hiro_amanokawa": {
        "character": ["hiro_amanokawa"],
        "trigger": ["hiro amanokawa, digimon"],
    },
    "classic_amy_rose": {
        "character": ["classic_amy_rose"],
        "trigger": ["classic amy rose, sonic the hedgehog \\(series\\)"],
    },
    "jake_the_dog": {
        "character": ["jake_the_dog"],
        "trigger": ["jake the dog, cartoon network"],
    },
    "tagg": {"character": ["tagg"], "trigger": ["tagg, the elder scrolls"]},
    "teryx_commodore": {
        "character": ["teryx_commodore"],
        "trigger": ["teryx commodore, mythology"],
    },
    "chubby_protagonist_(tas)": {
        "character": ["chubby_protagonist_(tas)"],
        "trigger": ["chubby protagonist \\(tas\\), lifewonders"],
    },
    "zen_(twokinds)": {
        "character": ["zen_(twokinds)"],
        "trigger": ["zen \\(twokinds\\), twokinds"],
    },
    "aleu_(balto)": {
        "character": ["aleu_(balto)"],
        "trigger": ["aleu \\(balto\\), universal studios"],
    },
    "sledge": {"character": ["sledge"], "trigger": ["sledge, mythology"]},
    "alex_(harmarist)": {
        "character": ["alex_(harmarist)"],
        "trigger": ["alex \\(harmarist\\), sheath and knife"],
    },
    "maggie_hudson": {
        "character": ["maggie_hudson"],
        "trigger": ["maggie hudson, christmas"],
    },
    "pear_butter_(mlp)": {
        "character": ["pear_butter_(mlp)"],
        "trigger": ["pear butter \\(mlp\\), my little pony"],
    },
    "young_link": {
        "character": ["young_link"],
        "trigger": ["young link, the legend of zelda"],
    },
    "vitani_(the_lion_king)": {
        "character": ["vitani_(the_lion_king)"],
        "trigger": ["vitani \\(the lion king\\), disney"],
    },
    "tany_(doneru)": {
        "character": ["tany_(doneru)"],
        "trigger": ["tany \\(doneru\\), mythology"],
    },
    "reed_(gvh)": {
        "character": ["reed_(gvh)"],
        "trigger": ["reed \\(gvh\\), goodbye volcano high"],
    },
    "flash_sentry_(mlp)": {
        "character": ["flash_sentry_(mlp)"],
        "trigger": ["flash sentry \\(mlp\\), my little pony"],
    },
    "leon_powalski": {
        "character": ["leon_powalski"],
        "trigger": ["leon powalski, star fox"],
    },
    "fink_(ok_k.o.!_lbh)": {
        "character": ["fink_(ok_k.o.!_lbh)"],
        "trigger": ["fink \\(ok k.o.! lbh\\), cartoon network"],
    },
    "rosa_(gvh)": {
        "character": ["rosa_(gvh)"],
        "trigger": ["rosa \\(gvh\\), goodbye volcano high"],
    },
    "yakko_warner": {
        "character": ["yakko_warner"],
        "trigger": ["yakko warner, warner brothers"],
    },
    "the_deadly_six": {
        "character": ["the_deadly_six"],
        "trigger": ["the deadly six, sonic the hedgehog \\(series\\)"],
    },
    "infinite_(sonic)": {
        "character": ["infinite_(sonic)"],
        "trigger": ["infinite \\(sonic\\), sonic the hedgehog \\(series\\)"],
    },
    "pipsqueak_(mlp)": {
        "character": ["pipsqueak_(mlp)"],
        "trigger": ["pipsqueak \\(mlp\\), my little pony"],
    },
    "alopex": {
        "character": ["alopex"],
        "trigger": ["alopex, teenage mutant ninja turtles"],
    },
    "bronwyn": {"character": ["bronwyn"], "trigger": ["bronwyn, cartoon network"]},
    "scourge_the_hedgehog": {
        "character": ["scourge_the_hedgehog"],
        "trigger": ["scourge the hedgehog, sonic the hedgehog \\(series\\)"],
    },
    "dixie_kong": {
        "character": ["dixie_kong"],
        "trigger": ["dixie kong, donkey kong \\(series\\)"],
    },
    "gabby_(mlp)": {
        "character": ["gabby_(mlp)"],
        "trigger": ["gabby \\(mlp\\), my little pony"],
    },
    "off_the_hook_(splatoon)": {
        "character": ["off_the_hook_(splatoon)"],
        "trigger": ["off the hook \\(splatoon\\), splatoon"],
    },
    "dale_(disney)": {
        "character": ["dale_(disney)"],
        "trigger": ["dale \\(disney\\), disney"],
    },
    "ero_(erobos)": {
        "character": ["ero_(erobos)"],
        "trigger": ["ero \\(erobos\\), nintendo"],
    },
    "windy_whistles_(mlp)": {
        "character": ["windy_whistles_(mlp)"],
        "trigger": ["windy whistles \\(mlp\\), my little pony"],
    },
    "masyunya_(vkontakte)": {
        "character": ["masyunya_(vkontakte)"],
        "trigger": ["masyunya \\(vkontakte\\), vkontakte"],
    },
    "bill_(beastars)": {
        "character": ["bill_(beastars)"],
        "trigger": ["bill \\(beastars\\), beastars"],
    },
    "king_ghidorah": {
        "character": ["king_ghidorah"],
        "trigger": ["king ghidorah, godzilla \\(series\\)"],
    },
    "0r0": {"character": ["0r0"], "trigger": ["0r0, my little pony"]},
    "rosie_(animal_crossing)": {
        "character": ["rosie_(animal_crossing)"],
        "trigger": ["rosie \\(animal crossing\\), animal crossing"],
    },
    "makoto_nanaya": {
        "character": ["makoto_nanaya"],
        "trigger": ["makoto nanaya, arc system works"],
    },
    "rip_(psy101)": {
        "character": ["rip_(psy101)"],
        "trigger": ["rip \\(psy101\\), disney"],
    },
    "dogday_(poppy_playtime)": {
        "character": ["dogday_(poppy_playtime)"],
        "trigger": ["dogday \\(poppy playtime\\), smiling critters"],
    },
    "kennen_(lol)": {
        "character": ["kennen_(lol)"],
        "trigger": ["kennen \\(lol\\), riot games"],
    },
    "kit_cloudkicker": {
        "character": ["kit_cloudkicker"],
        "trigger": ["kit cloudkicker, disney"],
    },
    "angela_cross": {
        "character": ["angela_cross"],
        "trigger": ["angela cross, sony corporation"],
    },
    "beast_(disney)": {
        "character": ["beast_(disney)"],
        "trigger": ["beast \\(disney\\), disney"],
    },
    "toy_bonnie_(psychojohn2)": {
        "character": ["toy_bonnie_(psychojohn2)"],
        "trigger": ["toy bonnie \\(psychojohn2\\), scottgames"],
    },
    "crusch_lulu": {
        "character": ["crusch_lulu"],
        "trigger": ["crusch lulu, overlord \\(series\\)"],
    },
    "gouhin_(beastars)": {
        "character": ["gouhin_(beastars)"],
        "trigger": ["gouhin \\(beastars\\), beastars"],
    },
    "master_viper": {
        "character": ["master_viper"],
        "trigger": ["master viper, kung fu panda"],
    },
    "gear_(foxgear)": {
        "character": ["gear_(foxgear)"],
        "trigger": ["gear \\(foxgear\\), mythology"],
    },
    "jessica_rabbit": {
        "character": ["jessica_rabbit"],
        "trigger": ["jessica rabbit, disney"],
    },
    "red_(pokemon)": {
        "character": ["red_(pokemon)"],
        "trigger": ["red \\(pokemon\\), pokemon"],
    },
    "panda_(we_bare_bears)": {
        "character": ["panda_(we_bare_bears)"],
        "trigger": ["panda \\(we bare bears\\), cartoon network"],
    },
    "piko_(simplifypm)": {
        "character": ["piko_(simplifypm)"],
        "trigger": ["piko \\(simplifypm\\), mythology"],
    },
    "fay_(star_fox)": {
        "character": ["fay_(star_fox)"],
        "trigger": ["fay \\(star fox\\), star fox"],
    },
    "twilight_sparkle_(eg)": {
        "character": ["twilight_sparkle_(eg)"],
        "trigger": ["twilight sparkle \\(eg\\), my little pony"],
    },
    "puzzle_(kadath)": {
        "character": ["puzzle_(kadath)"],
        "trigger": ["puzzle \\(kadath\\), patreon"],
    },
    "crazy_redd": {
        "character": ["crazy_redd"],
        "trigger": ["crazy redd, animal crossing"],
    },
    "gantu": {"character": ["gantu"], "trigger": ["gantu, disney"]},
    "cosmo_the_seedrian": {
        "character": ["cosmo_the_seedrian"],
        "trigger": ["cosmo the seedrian, sonic the hedgehog \\(series\\)"],
    },
    "velma_dinkley": {
        "character": ["velma_dinkley"],
        "trigger": ["velma dinkley, scooby-doo \\(series\\)"],
    },
    "mike_schmidt": {
        "character": ["mike_schmidt"],
        "trigger": ["mike schmidt, scottgames"],
    },
    "mabel_able": {
        "character": ["mabel_able"],
        "trigger": ["mabel able, animal crossing"],
    },
    "roflfox": {"character": ["roflfox"], "trigger": ["roflfox, nintendo"]},
    "officer_flint_(foretbwat)": {
        "character": ["officer_flint_(foretbwat)"],
        "trigger": ["officer flint \\(foretbwat\\), warfare machine"],
    },
    "remmy_cormo": {
        "character": ["remmy_cormo"],
        "trigger": ["remmy cormo, pack street"],
    },
    "wilykit": {"character": ["wilykit"], "trigger": ["wilykit, thundercats"]},
    "gloria_(pokemon)": {
        "character": ["gloria_(pokemon)"],
        "trigger": ["gloria \\(pokemon\\), pokemon"],
    },
    "timon": {"character": ["timon"], "trigger": ["timon, disney"]},
    "will_(harmarist)": {
        "character": ["will_(harmarist)"],
        "trigger": ["will \\(harmarist\\), sheath and knife"],
    },
    "macan_(tas)": {
        "character": ["macan_(tas)"],
        "trigger": ["macan \\(tas\\), lifewonders"],
    },
    "heart_(mad_rat_dead)": {
        "character": ["heart_(mad_rat_dead)"],
        "trigger": ["heart \\(mad rat dead\\), nippon ichi software"],
    },
    "nazrin": {"character": ["nazrin"], "trigger": ["nazrin, touhou"]},
    "zhali": {"character": ["zhali"], "trigger": ["zhali, mythology"]},
    "chip_(disney)": {
        "character": ["chip_(disney)"],
        "trigger": ["chip \\(disney\\), disney"],
    },
    "jake_clawson": {
        "character": ["jake_clawson"],
        "trigger": ["jake clawson, swat kats"],
    },
    "niko_(oneshot)": {
        "character": ["niko_(oneshot)"],
        "trigger": ["niko \\(oneshot\\), oneshot"],
    },
    "yorha_2b": {"character": ["yorha_2b"], "trigger": ["yorha 2b, platinumgames"]},
    "miia_(monster_musume)": {
        "character": ["miia_(monster_musume)"],
        "trigger": ["miia \\(monster musume\\), monster musume"],
    },
    "jess_(teckly)": {
        "character": ["jess_(teckly)"],
        "trigger": ["jess \\(teckly\\), mythology"],
    },
    "majin_android_21": {
        "character": ["majin_android_21"],
        "trigger": ["majin android 21, dragon ball"],
    },
    "tsukino_(monster_hunter_stories)": {
        "character": ["tsukino_(monster_hunter_stories)"],
        "trigger": [
            "tsukino \\(monster hunter stories\\), monster hunter stories 2: wings of ruin"
        ],
    },
    "grizzly_(we_bare_bears)": {
        "character": ["grizzly_(we_bare_bears)"],
        "trigger": ["grizzly \\(we bare bears\\), cartoon network"],
    },
    "bonnie_hopps": {
        "character": ["bonnie_hopps"],
        "trigger": ["bonnie hopps, disney"],
    },
    "perdita": {"character": ["perdita"], "trigger": ["perdita, disney"]},
    "milky_way_(flash_equestria)": {
        "character": ["milky_way_(flash_equestria)"],
        "trigger": ["milky way \\(flash equestria\\), my little pony"],
    },
    "espio_the_chameleon": {
        "character": ["espio_the_chameleon"],
        "trigger": ["espio the chameleon, sonic the hedgehog \\(series\\)"],
    },
    "kit_(kitsune_youkai)": {
        "character": ["kit_(kitsune_youkai)"],
        "trigger": ["kit \\(kitsune youkai\\), nintendo"],
    },
    "lightning_dust_(mlp)": {
        "character": ["lightning_dust_(mlp)"],
        "trigger": ["lightning dust \\(mlp\\), my little pony"],
    },
    "shippou_(inuyasha)": {
        "character": ["shippou_(inuyasha)"],
        "trigger": ["shippou \\(inuyasha\\), inuyasha"],
    },
    "panther_caroso": {
        "character": ["panther_caroso"],
        "trigger": ["panther caroso, star fox"],
    },
    "whitney_(animal_crossing)": {
        "character": ["whitney_(animal_crossing)"],
        "trigger": ["whitney \\(animal crossing\\), animal crossing"],
    },
    "hello_kitty_(character)": {
        "character": ["hello_kitty_(character)"],
        "trigger": ["hello kitty \\(character\\), hello kitty \\(series\\)"],
    },
    "kanga": {"character": ["kanga"], "trigger": ["kanga, disney"]},
    "lila_(kashiwagi_aki)": {
        "character": ["lila_(kashiwagi_aki)"],
        "trigger": ["lila \\(kashiwagi aki\\), the beast and his pet high school girl"],
    },
    "lanolin_the_sheep_(sonic)": {
        "character": ["lanolin_the_sheep_(sonic)"],
        "trigger": ["lanolin the sheep \\(sonic\\), sonic the hedgehog \\(series\\)"],
    },
    "goemon_(tas)": {
        "character": ["goemon_(tas)"],
        "trigger": ["goemon \\(tas\\), lifewonders"],
    },
    "asui_tsuyu": {
        "character": ["asui_tsuyu"],
        "trigger": ["asui tsuyu, my hero academia"],
    },
    "punchy_(animal_crossing)": {
        "character": ["punchy_(animal_crossing)"],
        "trigger": ["punchy \\(animal crossing\\), animal crossing"],
    },
    "peridot_(steven_universe)": {
        "character": ["peridot_(steven_universe)"],
        "trigger": ["peridot \\(steven universe\\), cartoon network"],
    },
    "warfare_renamon": {
        "character": ["warfare_renamon"],
        "trigger": ["warfare renamon, digimon"],
    },
    "katt_monroe": {"character": ["katt_monroe"], "trigger": ["katt monroe, star fox"]},
    "serena_(pokemon)": {
        "character": ["serena_(pokemon)"],
        "trigger": ["serena \\(pokemon\\), pokemon"],
    },
    "shino_(tas)": {
        "character": ["shino_(tas)"],
        "trigger": ["shino \\(tas\\), lifewonders"],
    },
    "pack_leader_highwire": {
        "character": ["pack_leader_highwire"],
        "trigger": ["pack leader highwire, fortnite"],
    },
    "catti_(deltarune)": {
        "character": ["catti_(deltarune)"],
        "trigger": ["catti \\(deltarune\\), undertale \\(series\\)"],
    },
    "carrot_top_(mlp)": {
        "character": ["carrot_top_(mlp)"],
        "trigger": ["carrot top \\(mlp\\), my little pony"],
    },
    "gummy_(mlp)": {
        "character": ["gummy_(mlp)"],
        "trigger": ["gummy \\(mlp\\), my little pony"],
    },
    "zinovy": {
        "character": ["zinovy"],
        "trigger": ["zinovy, the beast and his pet high school girl"],
    },
    "mrs._shy_(mlp)": {
        "character": ["mrs._shy_(mlp)"],
        "trigger": ["mrs. shy \\(mlp\\), my little pony"],
    },
    "noms_(nimzy)": {
        "character": ["noms_(nimzy)"],
        "trigger": ["noms \\(nimzy\\), christmas"],
    },
    "sue_sakamoto": {
        "character": ["sue_sakamoto"],
        "trigger": ["sue sakamoto, cave story"],
    },
    "cynthia_(pokemon)": {
        "character": ["cynthia_(pokemon)"],
        "trigger": ["cynthia \\(pokemon\\), pokemon"],
    },
    "reppy_(mlp)": {
        "character": ["reppy_(mlp)"],
        "trigger": ["reppy \\(mlp\\), my little pony"],
    },
    "calem_(pokemon)": {
        "character": ["calem_(pokemon)"],
        "trigger": ["calem \\(pokemon\\), pokemon"],
    },
    "pone_keith": {
        "character": ["pone_keith"],
        "trigger": ["pone keith, my little pony"],
    },
    "katrina_fowler": {
        "character": ["katrina_fowler"],
        "trigger": ["katrina fowler, patreon"],
    },
    "raven_hunt": {
        "character": ["raven_hunt"],
        "trigger": ["raven hunt, furafterdark"],
    },
    "mrs._brisby": {
        "character": ["mrs._brisby"],
        "trigger": ["mrs. brisby, don bluth"],
    },
    "gwen_tennyson": {
        "character": ["gwen_tennyson"],
        "trigger": ["gwen tennyson, cartoon network"],
    },
    "winnie_werewolf_(hotel_transylvania)": {
        "character": ["winnie_werewolf_(hotel_transylvania)"],
        "trigger": ["winnie werewolf \\(hotel transylvania\\), hotel transylvania"],
    },
    "akari_(pokemon)": {
        "character": ["akari_(pokemon)"],
        "trigger": ["akari \\(pokemon\\), pokemon"],
    },
    "minuette_(mlp)": {
        "character": ["minuette_(mlp)"],
        "trigger": ["minuette \\(mlp\\), my little pony"],
    },
    "buxbi_(character)": {
        "character": ["buxbi_(character)"],
        "trigger": ["buxbi \\(character\\), christmas"],
    },
    "narinder": {"character": ["narinder"], "trigger": ["narinder, cult of the lamb"]},
    "k.k._slider": {
        "character": ["k.k._slider"],
        "trigger": ["k.k. slider, animal crossing"],
    },
    "drum_bunker_dragon": {
        "character": ["drum_bunker_dragon"],
        "trigger": ["drum bunker dragon, future card buddyfight"],
    },
    "neptune_mereaux": {
        "character": ["neptune_mereaux"],
        "trigger": ["neptune mereaux, nintendo"],
    },
    "lop_(star_wars_visions)": {
        "character": ["lop_(star_wars_visions)"],
        "trigger": ["lop \\(star wars visions\\), star wars visions"],
    },
    "batman": {"character": ["batman"], "trigger": ["batman, dc comics"]},
    "ocellus_(mlp)": {
        "character": ["ocellus_(mlp)"],
        "trigger": ["ocellus \\(mlp\\), my little pony"],
    },
    "elza_(interspecies_reviewers)": {
        "character": ["elza_(interspecies_reviewers)"],
        "trigger": ["elza \\(interspecies reviewers\\), interspecies reviewers"],
    },
    "colleen_(road_rovers)": {
        "character": ["colleen_(road_rovers)"],
        "trigger": ["colleen \\(road rovers\\), road rovers"],
    },
    "bojack_horseman_(character)": {
        "character": ["bojack_horseman_(character)"],
        "trigger": ["bojack horseman \\(character\\), netflix"],
    },
    "daxter": {"character": ["daxter"], "trigger": ["daxter, jak and daxter"]},
    "dingodile": {
        "character": ["dingodile"],
        "trigger": ["dingodile, crash bandicoot \\(series\\)"],
    },
    "chip_(sonic)": {
        "character": ["chip_(sonic)"],
        "trigger": ["chip \\(sonic\\), sonic the hedgehog \\(series\\)"],
    },
    "yuni_hermit": {
        "character": ["yuni_hermit"],
        "trigger": ["yuni hermit, mythology"],
    },
    "skylar_fidchell": {
        "character": ["skylar_fidchell"],
        "trigger": ["skylar fidchell, if hell had a taste"],
    },
    "hazel_(shakotanbunny)": {
        "character": ["hazel_(shakotanbunny)"],
        "trigger": ["hazel \\(shakotanbunny\\), mythology"],
    },
    "rose_(mlp)": {
        "character": ["rose_(mlp)"],
        "trigger": ["rose \\(mlp\\), my little pony"],
    },
    "goat_lucifer_(helltaker)": {
        "character": ["goat_lucifer_(helltaker)"],
        "trigger": ["goat lucifer \\(helltaker\\), helltaker"],
    },
    "kounosuke_(morenatsu)": {
        "character": ["kounosuke_(morenatsu)"],
        "trigger": ["kounosuke \\(morenatsu\\), morenatsu"],
    },
    "protagonist_(live_a_hero)": {
        "character": ["protagonist_(live_a_hero)"],
        "trigger": ["protagonist \\(live a hero\\), lifewonders"],
    },
    "gwen_martin": {
        "character": ["gwen_martin"],
        "trigger": ["gwen martin, gwen geek"],
    },
    "bunsen": {"character": ["bunsen"], "trigger": ["bunsen, mythology"]},
    "purgy": {"character": ["purgy"], "trigger": ["purgy, mythology"]},
    "squidward_tentacles": {
        "character": ["squidward_tentacles"],
        "trigger": ["squidward tentacles, spongebob squarepants"],
    },
    "apollo_(animal_crossing)": {
        "character": ["apollo_(animal_crossing)"],
        "trigger": ["apollo \\(animal crossing\\), animal crossing"],
    },
    "kyubi_(yo-kai_watch)": {
        "character": ["kyubi_(yo-kai_watch)"],
        "trigger": ["kyubi \\(yo-kai watch\\), yo-kai watch"],
    },
    "medli": {"character": ["medli"], "trigger": ["medli, the legend of zelda"]},
    "cozy_glow_(mlp)": {
        "character": ["cozy_glow_(mlp)"],
        "trigger": ["cozy glow \\(mlp\\), my little pony"],
    },
    "leona_(aka)_little_one": {
        "character": ["leona_(aka)_little_one"],
        "trigger": ["leona \\(aka\\) little one, mythology"],
    },
    "olivia_halford": {
        "character": ["olivia_halford"],
        "trigger": ["olivia halford, cavemanon studios"],
    },
    "oscar_(fortnite)": {
        "character": ["oscar_(fortnite)"],
        "trigger": ["oscar \\(fortnite\\), fortnite"],
    },
    "princess_molestia": {
        "character": ["princess_molestia"],
        "trigger": ["princess molestia, my little pony"],
    },
    "easter_bunny": {
        "character": ["easter_bunny"],
        "trigger": ["easter bunny, easter"],
    },
    "werefox_(character)": {
        "character": ["werefox_(character)"],
        "trigger": ["werefox \\(character\\), mythology"],
    },
    "rory_(ceehaz)": {
        "character": ["rory_(ceehaz)"],
        "trigger": ["rory \\(ceehaz\\), dog knight rpg"],
    },
    "funtime_foxy_(fnafsl)": {
        "character": ["funtime_foxy_(fnafsl)"],
        "trigger": ["funtime foxy \\(fnafsl\\), scottgames"],
    },
    "boon_digges": {
        "character": ["boon_digges"],
        "trigger": ["boon digges, mythology"],
    },
    "roy_koopa": {"character": ["roy_koopa"], "trigger": ["roy koopa, mario bros"]},
    "ice_bear_(we_bare_bears)": {
        "character": ["ice_bear_(we_bare_bears)"],
        "trigger": ["ice bear \\(we bare bears\\), cartoon network"],
    },
    "tree_hugger_(mlp)": {
        "character": ["tree_hugger_(mlp)"],
        "trigger": ["tree hugger \\(mlp\\), my little pony"],
    },
    "cuddle_team_leader": {
        "character": ["cuddle_team_leader"],
        "trigger": ["cuddle team leader, fortnite"],
    },
    "courage_the_cowardly_dog_(character)": {
        "character": ["courage_the_cowardly_dog_(character)"],
        "trigger": ["courage the cowardly dog \\(character\\), cartoon network"],
    },
    "tiny_kong": {
        "character": ["tiny_kong"],
        "trigger": ["tiny kong, donkey kong \\(series\\)"],
    },
    "squid_sisters_(splatoon)": {
        "character": ["squid_sisters_(splatoon)"],
        "trigger": ["squid sisters \\(splatoon\\), splatoon"],
    },
    "kled_(lol)": {
        "character": ["kled_(lol)"],
        "trigger": ["kled \\(lol\\), riot games"],
    },
    "sea_salt": {"character": ["sea_salt"], "trigger": ["sea salt, the deep dark"]},
    "wendell_(fortnite)": {
        "character": ["wendell_(fortnite)"],
        "trigger": ["wendell \\(fortnite\\), fortnite"],
    },
    "everest_(paw_patrol)": {
        "character": ["everest_(paw_patrol)"],
        "trigger": ["everest \\(paw patrol\\), paw patrol"],
    },
    "nubless": {
        "character": ["nubless"],
        "trigger": ["nubless, how to train your dragon"],
    },
    "red_savarin": {
        "character": ["red_savarin"],
        "trigger": ["red savarin, solatorobo"],
    },
    "night_(dream_and_nightmare)": {
        "character": ["night_(dream_and_nightmare)"],
        "trigger": ["night \\(dream and nightmare\\), mythology"],
    },
    "pac-man": {"character": ["pac-man"], "trigger": ["pac-man, pac-man \\(series\\)"]},
    "toadette": {"character": ["toadette"], "trigger": ["toadette, nintendo"]},
    "toy_chica_(psychojohn2)": {
        "character": ["toy_chica_(psychojohn2)"],
        "trigger": ["toy chica \\(psychojohn2\\), scottgames"],
    },
    "spitey": {"character": ["spitey"], "trigger": ["spitey, mythology"]},
    "dudley_puppy": {
        "character": ["dudley_puppy"],
        "trigger": ["dudley puppy, t.u.f.f. puppy"],
    },
    "dr._voir": {"character": ["dr._voir"], "trigger": ["dr. voir, pokemon"]},
    "luck_(animancer)": {
        "character": ["luck_(animancer)"],
        "trigger": ["luck \\(animancer\\), square enix"],
    },
    "808_(hi-fi_rush)": {
        "character": ["808_(hi-fi_rush)"],
        "trigger": ["808 \\(hi-fi rush\\), hi-fi rush"],
    },
    "tommy_nook": {
        "character": ["tommy_nook"],
        "trigger": ["tommy nook, animal crossing"],
    },
    "fredina_(cally3d)": {
        "character": ["fredina_(cally3d)"],
        "trigger": ["fredina \\(cally3d\\), scottgames"],
    },
    "fionna_the_human": {
        "character": ["fionna_the_human"],
        "trigger": ["fionna the human, cartoon network"],
    },
    "likulau": {"character": ["likulau"], "trigger": ["likulau, studio klondike"]},
    "lin_hu": {"character": ["lin_hu"], "trigger": ["lin hu, studio klondike"]},
    "aak_(arknights)": {
        "character": ["aak_(arknights)"],
        "trigger": ["aak \\(arknights\\), studio montagne"],
    },
    "stellar_flare_(mlp)": {
        "character": ["stellar_flare_(mlp)"],
        "trigger": ["stellar flare \\(mlp\\), my little pony"],
    },
    "mrs._wilde": {"character": ["mrs._wilde"], "trigger": ["mrs. wilde, disney"]},
    "timmy_nook": {
        "character": ["timmy_nook"],
        "trigger": ["timmy nook, animal crossing"],
    },
    "tulin_(tloz)": {
        "character": ["tulin_(tloz)"],
        "trigger": ["tulin \\(tloz\\), the legend of zelda"],
    },
    "aisyah_zaskia_harnny": {
        "character": ["aisyah_zaskia_harnny"],
        "trigger": ["aisyah zaskia harnny, mythology"],
    },
    "selene_leni": {
        "character": ["selene_leni"],
        "trigger": ["selene leni, mythology"],
    },
    "zabivaka": {"character": ["zabivaka"], "trigger": ["zabivaka, fifa"]},
    "miranda_(wakfu)": {
        "character": ["miranda_(wakfu)"],
        "trigger": ["miranda \\(wakfu\\), ankama"],
    },
    "esix": {"character": ["esix"], "trigger": ["esix, e621"]},
    "sawyer_(cats_don't_dance)": {
        "character": ["sawyer_(cats_don't_dance)"],
        "trigger": ["sawyer \\(cats don't dance\\), warner brothers"],
    },
    "winnie_werewolf_(ghoul_school)": {
        "character": ["winnie_werewolf_(ghoul_school)"],
        "trigger": ["winnie werewolf \\(ghoul school\\), scooby-doo \\(series\\)"],
    },
    "hiccup_horrendous_haddock_iii": {
        "character": ["hiccup_horrendous_haddock_iii"],
        "trigger": ["hiccup horrendous haddock iii, how to train your dragon"],
    },
    "classic_sonic": {
        "character": ["classic_sonic"],
        "trigger": ["classic sonic, sonic the hedgehog \\(series\\)"],
    },
    "cuphead_(character)": {
        "character": ["cuphead_(character)"],
        "trigger": ["cuphead \\(character\\), cuphead \\(game\\)"],
    },
    "laura_(twokinds)": {
        "character": ["laura_(twokinds)"],
        "trigger": ["laura \\(twokinds\\), twokinds"],
    },
    "sonata_dusk_(eg)": {
        "character": ["sonata_dusk_(eg)"],
        "trigger": ["sonata dusk \\(eg\\), my little pony"],
    },
    "aurelion_sol_(lol)": {
        "character": ["aurelion_sol_(lol)"],
        "trigger": ["aurelion sol \\(lol\\), riot games"],
    },
    "wendell": {"character": ["wendell"], "trigger": ["wendell, pokemon"]},
    "azula_arktandr": {
        "character": ["azula_arktandr"],
        "trigger": ["azula arktandr, nintendo"],
    },
    "wakko_warner": {
        "character": ["wakko_warner"],
        "trigger": ["wakko warner, warner brothers"],
    },
    "rosa_(pokemon)": {
        "character": ["rosa_(pokemon)"],
        "trigger": ["rosa \\(pokemon\\), pokemon"],
    },
    "shin_(morenatsu)": {
        "character": ["shin_(morenatsu)"],
        "trigger": ["shin \\(morenatsu\\), morenatsu"],
    },
    "celio_(peritian)": {
        "character": ["celio_(peritian)"],
        "trigger": ["celio \\(peritian\\), tumblr"],
    },
    "lara_croft": {"character": ["lara_croft"], "trigger": ["lara croft, tomb raider"]},
    "zim_(invader_zim)": {
        "character": ["zim_(invader_zim)"],
        "trigger": ["zim \\(invader zim\\), invader zim"],
    },
    "pomni_(tadc)": {
        "character": ["pomni_(tadc)"],
        "trigger": ["pomni \\(tadc\\), the amazing digital circus"],
    },
    "megumi_bandicoot": {
        "character": ["megumi_bandicoot"],
        "trigger": ["megumi bandicoot, crash bandicoot \\(series\\)"],
    },
    "ruins_style_lucario": {
        "character": ["ruins_style_lucario"],
        "trigger": ["ruins style lucario, pokemon unite"],
    },
    "mirage_(disney)": {
        "character": ["mirage_(disney)"],
        "trigger": ["mirage \\(disney\\), disney"],
    },
    "gloria_the_hippopotamus": {
        "character": ["gloria_the_hippopotamus"],
        "trigger": ["gloria the hippopotamus, madagascar \\(series\\)"],
    },
    "suki_lane": {
        "character": ["suki_lane"],
        "trigger": ["suki lane, illumination entertainment"],
    },
    "ben_bigger": {"character": ["ben_bigger"], "trigger": ["ben bigger, mihoyo"]},
    "snips_(mlp)": {
        "character": ["snips_(mlp)"],
        "trigger": ["snips \\(mlp\\), my little pony"],
    },
    "kicks_(animal_crossing)": {
        "character": ["kicks_(animal_crossing)"],
        "trigger": ["kicks \\(animal crossing\\), animal crossing"],
    },
    "agyo_(tas)": {
        "character": ["agyo_(tas)"],
        "trigger": ["agyo \\(tas\\), lifewonders"],
    },
    "rei_(pokemon)": {
        "character": ["rei_(pokemon)"],
        "trigger": ["rei \\(pokemon\\), pokemon"],
    },
    "shiver_(splatoon)": {
        "character": ["shiver_(splatoon)"],
        "trigger": ["shiver \\(splatoon\\), splatoon"],
    },
    "aamon_(james_howard)": {
        "character": ["aamon_(james_howard)"],
        "trigger": ["aamon \\(james howard\\), subscribestar"],
    },
    "vergence": {"character": ["vergence"], "trigger": ["vergence, creative commons"]},
    "lt._fox_vixen": {
        "character": ["lt._fox_vixen"],
        "trigger": ["lt. fox vixen, squirrel and hedgehog"],
    },
    "hugtastic_pinkie_pie": {
        "character": ["hugtastic_pinkie_pie"],
        "trigger": ["hugtastic pinkie pie, my little pony"],
    },
    "vicar_amelia": {
        "character": ["vicar_amelia"],
        "trigger": ["vicar amelia, bloodborne"],
    },
    "arno_(peritian)": {
        "character": ["arno_(peritian)"],
        "trigger": ["arno \\(peritian\\), tumblr"],
    },
    "jacki_northstar": {
        "character": ["jacki_northstar"],
        "trigger": ["jacki northstar, mythology"],
    },
    "zoe_trent": {"character": ["zoe_trent"], "trigger": ["zoe trent, hasbro"]},
    "parappa": {"character": ["parappa"], "trigger": ["parappa, parappa the rapper"]},
    "sean_(senz)": {
        "character": ["sean_(senz)"],
        "trigger": ["sean \\(senz\\), patreon"],
    },
    "king_clawthorne": {
        "character": ["king_clawthorne"],
        "trigger": ["king clawthorne, disney"],
    },
    "lady_(lady_and_the_tramp)": {
        "character": ["lady_(lady_and_the_tramp)"],
        "trigger": ["lady \\(lady and the tramp\\), disney"],
    },
    "marilyn_(quotefox)": {
        "character": ["marilyn_(quotefox)"],
        "trigger": ["marilyn \\(quotefox\\), no nut november"],
    },
    "ippan_josei": {
        "character": ["ippan_josei"],
        "trigger": ["ippan josei, my hero academia"],
    },
    "interstellar_demon_stripper": {
        "character": ["interstellar_demon_stripper"],
        "trigger": ["interstellar demon stripper, cartoon network"],
    },
    "alty": {"character": ["alty"], "trigger": ["alty, nintendo"]},
    "gemma_polson": {
        "character": ["gemma_polson"],
        "trigger": ["gemma polson, pokemon"],
    },
    "dixie_(tfath)": {
        "character": ["dixie_(tfath)"],
        "trigger": ["dixie \\(tfath\\), disney"],
    },
    "mr._cake_(mlp)": {
        "character": ["mr._cake_(mlp)"],
        "trigger": ["mr. cake \\(mlp\\), my little pony"],
    },
    "survivor_(rain_world)": {
        "character": ["survivor_(rain_world)"],
        "trigger": ["survivor \\(rain world\\), videocult"],
    },
    "frye_(splatoon)": {
        "character": ["frye_(splatoon)"],
        "trigger": ["frye \\(splatoon\\), splatoon"],
    },
    "patricia_bunny": {
        "character": ["patricia_bunny"],
        "trigger": ["patricia bunny, warner brothers"],
    },
    "oleander_(tfh)": {
        "character": ["oleander_(tfh)"],
        "trigger": ["oleander \\(tfh\\), them's fightin' herds"],
    },
    "helia_peppercats": {
        "character": ["helia_peppercats"],
        "trigger": ["helia peppercats, mythology"],
    },
    "hikari_kamiya": {
        "character": ["hikari_kamiya"],
        "trigger": ["hikari kamiya, digimon"],
    },
    "dylan_(101_dalmatians)": {
        "character": ["dylan_(101_dalmatians)"],
        "trigger": ["dylan \\(101 dalmatians\\), disney"],
    },
    "zira_(the_lion_king)": {
        "character": ["zira_(the_lion_king)"],
        "trigger": ["zira \\(the lion king\\), disney"],
    },
    "unnamed_character": {
        "character": ["unnamed_character"],
        "trigger": ["unnamed character, mythology"],
    },
    "kitty_softpaws": {
        "character": ["kitty_softpaws"],
        "trigger": ["kitty softpaws, puss in boots \\(dreamworks\\)"],
    },
    "foxy_(cally3d)": {
        "character": ["foxy_(cally3d)"],
        "trigger": ["foxy \\(cally3d\\), scottgames"],
    },
    "caster_tamamo-no-mae": {
        "character": ["caster_tamamo-no-mae"],
        "trigger": ["caster tamamo-no-mae, type-moon"],
    },
    "sybil_(pseudoregalia)": {
        "character": ["sybil_(pseudoregalia)"],
        "trigger": ["sybil \\(pseudoregalia\\), pseudoregalia"],
    },
    "flitter_(mlp)": {
        "character": ["flitter_(mlp)"],
        "trigger": ["flitter \\(mlp\\), my little pony"],
    },
    "gawr_gura": {"character": ["gawr_gura"], "trigger": ["gawr gura, hololive"]},
    "kogenta_(onmyou_taisenki)": {
        "character": ["kogenta_(onmyou_taisenki)"],
        "trigger": ["kogenta \\(onmyou taisenki\\), onmyou taisenki"],
    },
    "zeena": {
        "character": ["zeena"],
        "trigger": ["zeena, sonic the hedgehog \\(series\\)"],
    },
    "pumkat": {"character": ["pumkat"], "trigger": ["pumkat, halloween"]},
    "tadano_(aggretsuko)": {
        "character": ["tadano_(aggretsuko)"],
        "trigger": ["tadano \\(aggretsuko\\), sanrio"],
    },
    "dinky_hooves_(mlp)": {
        "character": ["dinky_hooves_(mlp)"],
        "trigger": ["dinky hooves \\(mlp\\), my little pony"],
    },
    "meru_(merunyaa)": {
        "character": ["meru_(merunyaa)"],
        "trigger": ["meru \\(merunyaa\\), nintendo"],
    },
    "big_man_(splatoon)": {
        "character": ["big_man_(splatoon)"],
        "trigger": ["big man \\(splatoon\\), splatoon"],
    },
    "julie_bruin": {
        "character": ["julie_bruin"],
        "trigger": ["julie bruin, warner brothers"],
    },
    "scout_(team_fortress_2)": {
        "character": ["scout_(team_fortress_2)"],
        "trigger": ["scout \\(team fortress 2\\), team fortress 2"],
    },
    "doraemon_(character)": {
        "character": ["doraemon_(character)"],
        "trigger": ["doraemon \\(character\\), doraemon"],
    },
    "catra_(she-ra)": {
        "character": ["catra_(she-ra)"],
        "trigger": ["catra \\(she-ra\\), she-ra \\(copyright\\)"],
    },
    "bast": {"character": ["bast"], "trigger": ["bast, egyptian mythology"]},
    "leggy_lamb": {
        "character": ["leggy_lamb"],
        "trigger": ["leggy lamb, metro-goldwyn-mayer"],
    },
    "michelle_(dashboom)": {
        "character": ["michelle_(dashboom)"],
        "trigger": ["michelle \\(dashboom\\), nintendo"],
    },
    "mercenary_(grimoire_of_zero)": {
        "character": ["mercenary_(grimoire_of_zero)"],
        "trigger": ["mercenary \\(grimoire of zero\\), grimoire of zero"],
    },
    "pavita_pechugona": {
        "character": ["pavita_pechugona"],
        "trigger": ["pavita pechugona, la pavita pechugona"],
    },
    "tama-tama": {"character": ["tama-tama"], "trigger": ["tama-tama, prostokvashino"]},
    "takato_matsuki": {
        "character": ["takato_matsuki"],
        "trigger": ["takato matsuki, digimon"],
    },
    "lovetaste_chica": {
        "character": ["lovetaste_chica"],
        "trigger": ["lovetaste chica, five nights at freddy's 2"],
    },
    "pyron": {"character": ["pyron"], "trigger": ["pyron, mythology"]},
    "centorea_shianus_(monster_musume)": {
        "character": ["centorea_shianus_(monster_musume)"],
        "trigger": ["centorea shianus \\(monster musume\\), monster musume"],
    },
    "kae_esrial": {"character": ["kae_esrial"], "trigger": ["kae esrial, mythology"]},
    "izuku_midoriya": {
        "character": ["izuku_midoriya"],
        "trigger": ["izuku midoriya, my hero academia"],
    },
    "urbosa": {"character": ["urbosa"], "trigger": ["urbosa, breath of the wild"]},
    "eda_clawthorne": {
        "character": ["eda_clawthorne"],
        "trigger": ["eda clawthorne, disney"],
    },
    "chen_(touhou)": {
        "character": ["chen_(touhou)"],
        "trigger": ["chen \\(touhou\\), touhou"],
    },
    "margret_stalizburg": {
        "character": ["margret_stalizburg"],
        "trigger": ["margret stalizburg, mythology"],
    },
    "goku": {"character": ["goku"], "trigger": ["goku, dragon ball"]},
    "jen_(vf)": {"character": ["jen_(vf)"], "trigger": ["jen \\(vf\\), pokemon"]},
    "vibri": {"character": ["vibri"], "trigger": ["vibri, vib-ribbon"]},
    "simon_seville": {
        "character": ["simon_seville"],
        "trigger": ["simon seville, alvin and the chipmunks"],
    },
    "paprika_paca_(tfh)": {
        "character": ["paprika_paca_(tfh)"],
        "trigger": ["paprika paca \\(tfh\\), them's fightin' herds"],
    },
    "ornn_(lol)": {
        "character": ["ornn_(lol)"],
        "trigger": ["ornn \\(lol\\), riot games"],
    },
    "chico_(fuel)": {
        "character": ["chico_(fuel)"],
        "trigger": ["chico \\(fuel\\), disney"],
    },
    "louie_duck": {"character": ["louie_duck"], "trigger": ["louie duck, disney"]},
    "stu_hopps": {"character": ["stu_hopps"], "trigger": ["stu hopps, disney"]},
    "mia_(world_flipper)": {
        "character": ["mia_(world_flipper)"],
        "trigger": ["mia \\(world flipper\\), cygames"],
    },
    "pip_(paladins)": {
        "character": ["pip_(paladins)"],
        "trigger": ["pip \\(paladins\\), paladins \\(game\\)"],
    },
    "skylar_zero": {
        "character": ["skylar_zero"],
        "trigger": ["skylar zero, mythology"],
    },
    "shino_(animal_crossing)": {
        "character": ["shino_(animal_crossing)"],
        "trigger": ["shino \\(animal crossing\\), animal crossing"],
    },
    "servo": {"character": ["servo"], "trigger": ["servo, mythology"]},
    "metal_sonic": {
        "character": ["metal_sonic"],
        "trigger": ["metal sonic, sonic the hedgehog \\(series\\)"],
    },
    "rita_(jungledyret)": {
        "character": ["rita_(jungledyret)"],
        "trigger": ["rita \\(jungledyret\\), jungledyret hugo"],
    },
    "daisy_duck": {"character": ["daisy_duck"], "trigger": ["daisy duck, disney"]},
    "kanna_(blaster_master)": {
        "character": ["kanna_(blaster_master)"],
        "trigger": ["kanna \\(blaster master\\), blaster master"],
    },
    "aisha_clanclan": {
        "character": ["aisha_clanclan"],
        "trigger": ["aisha clanclan, outlaw star"],
    },
    "dakka": {"character": ["dakka"], "trigger": ["dakka, halloween"]},
    "karnal_(karnaltiger)": {
        "character": ["karnal_(karnaltiger)"],
        "trigger": ["karnal \\(karnaltiger\\), christmas"],
    },
    "penelope_(rainbowscreen)": {
        "character": ["penelope_(rainbowscreen)"],
        "trigger": ["penelope \\(rainbowscreen\\), mythology"],
    },
    "gervic_(vju79)": {
        "character": ["gervic_(vju79)"],
        "trigger": ["gervic \\(vju79\\), mythology"],
    },
    "tinker_bell_(disney)": {
        "character": ["tinker_bell_(disney)"],
        "trigger": ["tinker bell \\(disney\\), peter pan"],
    },
    "vixavil_hayden": {
        "character": ["vixavil_hayden"],
        "trigger": ["vixavil hayden, tale of tails"],
    },
    "lillie_(pokemon)": {
        "character": ["lillie_(pokemon)"],
        "trigger": ["lillie \\(pokemon\\), pokemon"],
    },
    "flaky_(htf)": {
        "character": ["flaky_(htf)"],
        "trigger": ["flaky \\(htf\\), happy tree friends"],
    },
    "arizona_cow_(tfh)": {
        "character": ["arizona_cow_(tfh)"],
        "trigger": ["arizona cow \\(tfh\\), them's fightin' herds"],
    },
    "shu-chi": {"character": ["shu-chi"], "trigger": ["shu-chi, studio klondike"]},
    "d.va_(overwatch)": {
        "character": ["d.va_(overwatch)"],
        "trigger": ["d.va \\(overwatch\\), overwatch"],
    },
    "urdnot_wrex": {
        "character": ["urdnot_wrex"],
        "trigger": ["urdnot wrex, mass effect"],
    },
    "mike_(twokinds)": {
        "character": ["mike_(twokinds)"],
        "trigger": ["mike \\(twokinds\\), twokinds"],
    },
    "muko": {"character": ["muko"], "trigger": ["muko, furryfight chronicles"]},
    "fwench_fwy_(chikn_nuggit)": {
        "character": ["fwench_fwy_(chikn_nuggit)"],
        "trigger": ["fwench fwy \\(chikn nuggit\\), chikn nuggit"],
    },
    "granny_smith_(mlp)": {
        "character": ["granny_smith_(mlp)"],
        "trigger": ["granny smith \\(mlp\\), my little pony"],
    },
    "milachu": {"character": ["milachu"], "trigger": ["milachu, pokemon"]},
    "spinel_(sepiruth)": {
        "character": ["spinel_(sepiruth)"],
        "trigger": ["spinel \\(sepiruth\\), mythology"],
    },
    "laverne_(sssonic2)": {
        "character": ["laverne_(sssonic2)"],
        "trigger": ["laverne \\(sssonic2\\), nintendo"],
    },
    "hybrid_(fortnite)": {
        "character": ["hybrid_(fortnite)"],
        "trigger": ["hybrid \\(fortnite\\), fortnite"],
    },
    "sora_(kingdom_hearts)": {
        "character": ["sora_(kingdom_hearts)"],
        "trigger": ["sora \\(kingdom hearts\\), kingdom hearts"],
    },
    "fluffle_puff": {
        "character": ["fluffle_puff"],
        "trigger": ["fluffle puff, my little pony"],
    },
    "renimpmon": {"character": ["renimpmon"], "trigger": ["renimpmon, digimon"]},
    "chikn_nuggit_(chikn_nuggit)": {
        "character": ["chikn_nuggit_(chikn_nuggit)"],
        "trigger": ["chikn nuggit \\(chikn nuggit\\), chikn nuggit"],
    },
    "olivia_(animal_crossing)": {
        "character": ["olivia_(animal_crossing)"],
        "trigger": ["olivia \\(animal crossing\\), animal crossing"],
    },
    "scp-682": {"character": ["scp-682"], "trigger": ["scp-682, scp foundation"]},
    "keeshee": {"character": ["keeshee"], "trigger": ["keeshee, christmas"]},
    "duck_hunt_dog": {
        "character": ["duck_hunt_dog"],
        "trigger": ["duck hunt dog, duck hunt"],
    },
    "shade_the_echidna": {
        "character": ["shade_the_echidna"],
        "trigger": ["shade the echidna, sonic chronicles: the dark brotherhood"],
    },
    "jack_frost_(megami_tensei)": {
        "character": ["jack_frost_(megami_tensei)"],
        "trigger": ["jack frost \\(megami tensei\\), sega"],
    },
    "etheras": {
        "character": ["etheras"],
        "trigger": ["etheras, etheras \\(copyright\\)"],
    },
    "patricia_mac_sionnach": {
        "character": ["patricia_mac_sionnach"],
        "trigger": ["patricia mac sionnach, fiat"],
    },
    "excellia_(coc)": {
        "character": ["excellia_(coc)"],
        "trigger": ["excellia \\(coc\\), corruption of champions"],
    },
    "alejandra_coldthorn": {
        "character": ["alejandra_coldthorn"],
        "trigger": ["alejandra coldthorn, las lindas"],
    },
    "tatsuki_(morenatsu)": {
        "character": ["tatsuki_(morenatsu)"],
        "trigger": ["tatsuki \\(morenatsu\\), morenatsu"],
    },
    "glitchtrap": {"character": ["glitchtrap"], "trigger": ["glitchtrap, scottgames"]},
    "auroth_the_winter_wyvern": {
        "character": ["auroth_the_winter_wyvern"],
        "trigger": ["auroth the winter wyvern, dota"],
    },
    "tohru_(dragon_maid)": {
        "character": ["tohru_(dragon_maid)"],
        "trigger": ["tohru \\(dragon maid\\), miss kobayashi's dragon maid"],
    },
    "krampus_(tas)": {
        "character": ["krampus_(tas)"],
        "trigger": ["krampus \\(tas\\), lifewonders"],
    },
    "sierra_(mana)": {
        "character": ["sierra_(mana)"],
        "trigger": ["sierra \\(mana\\), square enix"],
    },
    "mipha": {"character": ["mipha"], "trigger": ["mipha, breath of the wild"]},
    "bonnie_(cally3d)": {
        "character": ["bonnie_(cally3d)"],
        "trigger": ["bonnie \\(cally3d\\), scottgames"],
    },
    "harley_quinn": {
        "character": ["harley_quinn"],
        "trigger": ["harley quinn, dc comics"],
    },
    "marshall_(paw_patrol)": {
        "character": ["marshall_(paw_patrol)"],
        "trigger": ["marshall \\(paw patrol\\), paw patrol"],
    },
    "mabel_(cherrikissu)": {
        "character": ["mabel_(cherrikissu)"],
        "trigger": ["mabel \\(cherrikissu\\), nintendo"],
    },
    "artik_ninetails": {
        "character": ["artik_ninetails"],
        "trigger": ["artik ninetails, mythology"],
    },
    "fang_the_weavile": {
        "character": ["fang_the_weavile"],
        "trigger": ["fang the weavile, pokemon"],
    },
    "chilli_heeler": {
        "character": ["chilli_heeler"],
        "trigger": ["chilli heeler, bluey \\(series\\)"],
    },
    "burgerpants": {
        "character": ["burgerpants"],
        "trigger": ["burgerpants, undertale \\(series\\)"],
    },
    "jessie_(team_rocket)": {
        "character": ["jessie_(team_rocket)"],
        "trigger": ["jessie \\(team rocket\\), team rocket"],
    },
    "kamek": {"character": ["kamek"], "trigger": ["kamek, mario bros"]},
    "jewel_(rio)": {
        "character": ["jewel_(rio)"],
        "trigger": ["jewel \\(rio\\), blue sky studios"],
    },
    "spoiled_rich_(mlp)": {
        "character": ["spoiled_rich_(mlp)"],
        "trigger": ["spoiled rich \\(mlp\\), my little pony"],
    },
    "warfare_rouge": {
        "character": ["warfare_rouge"],
        "trigger": ["warfare rouge, sonic the hedgehog \\(series\\)"],
    },
    "jubei_(blazblue)": {
        "character": ["jubei_(blazblue)"],
        "trigger": ["jubei \\(blazblue\\), arc system works"],
    },
    "etis": {"character": ["etis"], "trigger": ["etis, mythology"]},
    "gregory_(fnaf)": {
        "character": ["gregory_(fnaf)"],
        "trigger": ["gregory \\(fnaf\\), five nights at freddy's: security breach"],
    },
    "custom_character_(sonic_forces)": {
        "character": ["custom_character_(sonic_forces)"],
        "trigger": [
            "custom character \\(sonic forces\\), sonic the hedgehog \\(series\\)"
        ],
    },
    "jumba_jookiba": {
        "character": ["jumba_jookiba"],
        "trigger": ["jumba jookiba, disney"],
    },
    "mega_man_(character)": {
        "character": ["mega_man_(character)"],
        "trigger": ["mega man \\(character\\), mega man \\(series\\)"],
    },
    "klodette": {"character": ["klodette"], "trigger": ["klodette, my little pony"]},
    "leodore_lionheart": {
        "character": ["leodore_lionheart"],
        "trigger": ["leodore lionheart, disney"],
    },
    "flippy_(htf)": {
        "character": ["flippy_(htf)"],
        "trigger": ["flippy \\(htf\\), happy tree friends"],
    },
    "rory_kenneigh": {
        "character": ["rory_kenneigh"],
        "trigger": ["rory kenneigh, my little pony"],
    },
    "avocato": {"character": ["avocato"], "trigger": ["avocato, final space"]},
    "dasha_(petruz)": {
        "character": ["dasha_(petruz)"],
        "trigger": ["dasha \\(petruz\\), petruz \\(copyright\\)"],
    },
    "synge": {"character": ["synge"], "trigger": ["synge, pokemon"]},
    "adagio_dazzle_(eg)": {
        "character": ["adagio_dazzle_(eg)"],
        "trigger": ["adagio dazzle \\(eg\\), my little pony"],
    },
    "shennong_(tas)": {
        "character": ["shennong_(tas)"],
        "trigger": ["shennong \\(tas\\), lifewonders"],
    },
    "mikhaila_kirov": {
        "character": ["mikhaila_kirov"],
        "trigger": ["mikhaila kirov, patreon"],
    },
    "slark_the_nightcrawler": {
        "character": ["slark_the_nightcrawler"],
        "trigger": ["slark the nightcrawler, dota"],
    },
    "withered_bonnie_(fnaf)": {
        "character": ["withered_bonnie_(fnaf)"],
        "trigger": ["withered bonnie \\(fnaf\\), five nights at freddy's 2"],
    },
    "liru_(magical_pokaan)": {
        "character": ["liru_(magical_pokaan)"],
        "trigger": ["liru \\(magical pokaan\\), magical pokaan"],
    },
    "loli_dragon_(berseepon09)": {
        "character": ["loli_dragon_(berseepon09)"],
        "trigger": ["loli dragon \\(berseepon09\\), mythology"],
    },
    "sabrith_ebonclaw": {
        "character": ["sabrith_ebonclaw"],
        "trigger": ["sabrith ebonclaw, square enix"],
    },
    "twink_protagonist_(tas)": {
        "character": ["twink_protagonist_(tas)"],
        "trigger": ["twink protagonist \\(tas\\), lifewonders"],
    },
    "berri": {"character": ["berri"], "trigger": ["berri, conker's bad fur day"]},
    "russell_(castbound)": {
        "character": ["russell_(castbound)"],
        "trigger": ["russell \\(castbound\\), mythology"],
    },
    "florian_(pokemon)": {
        "character": ["florian_(pokemon)"],
        "trigger": ["florian \\(pokemon\\), pokemon"],
    },
    "rumble_(lol)": {
        "character": ["rumble_(lol)"],
        "trigger": ["rumble \\(lol\\), riot games"],
    },
    "chase_(paw_patrol)": {
        "character": ["chase_(paw_patrol)"],
        "trigger": ["chase \\(paw patrol\\), paw patrol"],
    },
    "ms._tarantula_(the_bad_guys)": {
        "character": ["ms._tarantula_(the_bad_guys)"],
        "trigger": ["ms. tarantula \\(the bad guys\\), the bad guys"],
    },
    "zaire_(nightdancer)": {
        "character": ["zaire_(nightdancer)"],
        "trigger": ["zaire \\(nightdancer\\), mythology"],
    },
    "hervy": {"character": ["hervy"], "trigger": ["hervy, mythology"]},
    "doe_(alfa995)": {
        "character": ["doe_(alfa995)"],
        "trigger": ["doe \\(alfa995\\), patreon"],
    },
    "hadou_(satsui-n0-had0u)": {
        "character": ["hadou_(satsui-n0-had0u)"],
        "trigger": ["hadou \\(satsui-n0-had0u\\), nintendo"],
    },
    "leo_(red_earth)": {
        "character": ["leo_(red_earth)"],
        "trigger": ["leo \\(red earth\\), red earth"],
    },
    "cake_the_cat": {
        "character": ["cake_the_cat"],
        "trigger": ["cake the cat, cartoon network"],
    },
    "yama_the_dorumon": {
        "character": ["yama_the_dorumon"],
        "trigger": ["yama the dorumon, digimon"],
    },
    "wile_e._coyote": {
        "character": ["wile_e._coyote"],
        "trigger": ["wile e. coyote, warner brothers"],
    },
    "amon_(atrolux)": {
        "character": ["amon_(atrolux)"],
        "trigger": ["amon \\(atrolux\\), patreon"],
    },
    "shikabane_(aggretsuko)": {
        "character": ["shikabane_(aggretsuko)"],
        "trigger": ["shikabane \\(aggretsuko\\), sanrio"],
    },
    "sabrina_(sabrina_online)": {
        "character": ["sabrina_(sabrina_online)"],
        "trigger": ["sabrina \\(sabrina online\\), sabrina online"],
    },
    "lien-da": {
        "character": ["lien-da"],
        "trigger": ["lien-da, sonic the hedgehog \\(series\\)"],
    },
    "tezcatlipoca_(tas)": {
        "character": ["tezcatlipoca_(tas)"],
        "trigger": ["tezcatlipoca \\(tas\\), lifewonders"],
    },
    "serah_(black-kitten)": {
        "character": ["serah_(black-kitten)"],
        "trigger": ["serah \\(black-kitten\\), christmas"],
    },
    "victor_johansen": {
        "character": ["victor_johansen"],
        "trigger": ["victor johansen, mythology"],
    },
    "vaggie_(hazbin_hotel)": {
        "character": ["vaggie_(hazbin_hotel)"],
        "trigger": ["vaggie \\(hazbin hotel\\), hazbin hotel"],
    },
    "linna_auriandi_(character)": {
        "character": ["linna_auriandi_(character)"],
        "trigger": ["linna auriandi \\(character\\), mythology"],
    },
    "meta_knight": {
        "character": ["meta_knight"],
        "trigger": ["meta knight, kirby \\(series\\)"],
    },
    "mercy_(overwatch)": {
        "character": ["mercy_(overwatch)"],
        "trigger": ["mercy \\(overwatch\\), overwatch"],
    },
    "solar_flare_(pvz)": {
        "character": ["solar_flare_(pvz)"],
        "trigger": ["solar flare \\(pvz\\), plants vs. zombies heroes"],
    },
    "quill-weave": {
        "character": ["quill-weave"],
        "trigger": ["quill-weave, the elder scrolls"],
    },
    "the_shark_(changed)": {
        "character": ["the_shark_(changed)"],
        "trigger": ["the shark \\(changed\\), changed \\(video game\\)"],
    },
    "ganyu_(genshin_impact)": {
        "character": ["ganyu_(genshin_impact)"],
        "trigger": ["ganyu \\(genshin impact\\), mihoyo"],
    },
    "king_kazma": {"character": ["king_kazma"], "trigger": ["king kazma, summer wars"]},
    "hiroyuki_(morenatsu)": {
        "character": ["hiroyuki_(morenatsu)"],
        "trigger": ["hiroyuki \\(morenatsu\\), morenatsu"],
    },
    "tina_(james_howard)": {
        "character": ["tina_(james_howard)"],
        "trigger": ["tina \\(james howard\\), patreon"],
    },
    "iggy_koopa": {"character": ["iggy_koopa"], "trigger": ["iggy koopa, mario bros"]},
    "elh_melizee": {
        "character": ["elh_melizee"],
        "trigger": ["elh melizee, solatorobo"],
    },
    "monique_pussycat": {
        "character": ["monique_pussycat"],
        "trigger": ["monique pussycat, super fuck friends"],
    },
    "corrupt_cynder": {
        "character": ["corrupt_cynder"],
        "trigger": ["corrupt cynder, spyro the dragon"],
    },
    "alistar_(lol)": {
        "character": ["alistar_(lol)"],
        "trigger": ["alistar \\(lol\\), riot games"],
    },
    "princess_carolyn": {
        "character": ["princess_carolyn"],
        "trigger": ["princess carolyn, netflix"],
    },
    "webby_vanderquack": {
        "character": ["webby_vanderquack"],
        "trigger": ["webby vanderquack, disney"],
    },
    "soups_(superiorfox)": {
        "character": ["soups_(superiorfox)"],
        "trigger": ["soups \\(superiorfox\\), nintendo"],
    },
    "yoshi_(character)": {
        "character": ["yoshi_(character)"],
        "trigger": ["yoshi \\(character\\), nintendo"],
    },
    "ludwig_von_koopa": {
        "character": ["ludwig_von_koopa"],
        "trigger": ["ludwig von koopa, mario bros"],
    },
    "clank_(ratchet_and_clank)": {
        "character": ["clank_(ratchet_and_clank)"],
        "trigger": ["clank \\(ratchet and clank\\), sony corporation"],
    },
    "jeffybunny": {"character": ["jeffybunny"], "trigger": ["jeffybunny, nintendo"]},
    "chica_(cally3d)": {
        "character": ["chica_(cally3d)"],
        "trigger": ["chica \\(cally3d\\), scottgames"],
    },
    "lady_bow": {"character": ["lady_bow"], "trigger": ["lady bow, mario bros"]},
    "neeko_(lol)": {
        "character": ["neeko_(lol)"],
        "trigger": ["neeko \\(lol\\), riot games"],
    },
    "accelo_(character)": {
        "character": ["accelo_(character)"],
        "trigger": ["accelo \\(character\\), pokemon"],
    },
    "cthulhu": {"character": ["cthulhu"], "trigger": ["cthulhu, cthulhu mythos"]},
    "diamond_(kadath)": {
        "character": ["diamond_(kadath)"],
        "trigger": ["diamond \\(kadath\\), patreon"],
    },
    "jasper_(family_guy)": {
        "character": ["jasper_(family_guy)"],
        "trigger": ["jasper \\(family guy\\), family guy"],
    },
    "littlepip": {"character": ["littlepip"], "trigger": ["littlepip, my little pony"]},
    "jean_(minecraft)": {
        "character": ["jean_(minecraft)"],
        "trigger": ["jean \\(minecraft\\), microsoft"],
    },
    "silly_cat_(mauzymice)": {
        "character": ["silly_cat_(mauzymice)"],
        "trigger": ["silly cat \\(mauzymice\\), boy kisser \\(meme\\)"],
    },
    "prince_blueblood_(mlp)": {
        "character": ["prince_blueblood_(mlp)"],
        "trigger": ["prince blueblood \\(mlp\\), my little pony"],
    },
    "death_(personification)": {
        "character": ["death_(personification)"],
        "trigger": ["death \\(personification\\), loving reaper"],
    },
    "monomasa": {"character": ["monomasa"], "trigger": ["monomasa, lifewonders"]},
    "mr._shark_(the_bad_guys)": {
        "character": ["mr._shark_(the_bad_guys)"],
        "trigger": ["mr. shark \\(the bad guys\\), the bad guys"],
    },
    "lifts-her-tail": {
        "character": ["lifts-her-tail"],
        "trigger": ["lifts-her-tail, the elder scrolls"],
    },
    "rocky_rickaby": {
        "character": ["rocky_rickaby"],
        "trigger": ["rocky rickaby, lackadaisy"],
    },
    "carrie_krueger": {
        "character": ["carrie_krueger"],
        "trigger": ["carrie krueger, the amazing world of gumball"],
    },
    "robbie_(rotten_robbie)": {
        "character": ["robbie_(rotten_robbie)"],
        "trigger": ["robbie \\(rotten robbie\\), truly \\(drink\\)"],
    },
    "warfare_fox": {"character": ["warfare_fox"], "trigger": ["warfare fox, star fox"]},
    "general_yunan": {
        "character": ["general_yunan"],
        "trigger": ["general yunan, disney"],
    },
    "panini_(chowder)": {
        "character": ["panini_(chowder)"],
        "trigger": ["panini \\(chowder\\), cartoon network"],
    },
    "mihari": {"character": ["mihari"], "trigger": ["mihari, christmas"]},
    "pearl_(steven_universe)": {
        "character": ["pearl_(steven_universe)"],
        "trigger": ["pearl \\(steven universe\\), cartoon network"],
    },
    "kai_yun-jun": {
        "character": ["kai_yun-jun"],
        "trigger": ["kai yun-jun, mythology"],
    },
    "wanda_(one_piece)": {
        "character": ["wanda_(one_piece)"],
        "trigger": ["wanda \\(one piece\\), one piece"],
    },
    "alastor_(hazbin_hotel)": {
        "character": ["alastor_(hazbin_hotel)"],
        "trigger": ["alastor \\(hazbin hotel\\), hazbin hotel"],
    },
    "debidebi_debiru": {
        "character": ["debidebi_debiru"],
        "trigger": ["debidebi debiru, vtuber"],
    },
    "etna_(disgaea)": {
        "character": ["etna_(disgaea)"],
        "trigger": ["etna \\(disgaea\\), nippon ichi software"],
    },
    "pound_cake_(mlp)": {
        "character": ["pound_cake_(mlp)"],
        "trigger": ["pound cake \\(mlp\\), my little pony"],
    },
    "duke_weaselton": {
        "character": ["duke_weaselton"],
        "trigger": ["duke weaselton, disney"],
    },
    "jonty": {"character": ["jonty"], "trigger": ["jonty, a story with a known end"]},
    "geecku": {"character": ["geecku"], "trigger": ["geecku, las lindas"]},
    "rainstorm_(marefurryfan)": {
        "character": ["rainstorm_(marefurryfan)"],
        "trigger": ["rainstorm \\(marefurryfan\\), disney"],
    },
    "cookie_crumbles_(mlp)": {
        "character": ["cookie_crumbles_(mlp)"],
        "trigger": ["cookie crumbles \\(mlp\\), my little pony"],
    },
    "kima_(kimacats)": {
        "character": ["kima_(kimacats)"],
        "trigger": ["kima \\(kimacats\\), twitter"],
    },
    "cheetara": {"character": ["cheetara"], "trigger": ["cheetara, thundercats"]},
    "mia_moretti": {
        "character": ["mia_moretti"],
        "trigger": ["mia moretti, cavemanon studios"],
    },
    "spider-man_(character)": {
        "character": ["spider-man_(character)"],
        "trigger": ["spider-man \\(character\\), spider-man \\(series\\)"],
    },
    "mountain_(arknights)": {
        "character": ["mountain_(arknights)"],
        "trigger": ["mountain \\(arknights\\), studio montagne"],
    },
    "pj_(goof_troop)": {
        "character": ["pj_(goof_troop)"],
        "trigger": ["pj \\(goof troop\\), disney"],
    },
    "pumpkin_cake_(mlp)": {
        "character": ["pumpkin_cake_(mlp)"],
        "trigger": ["pumpkin cake \\(mlp\\), my little pony"],
    },
    "selina_zifer": {
        "character": ["selina_zifer"],
        "trigger": ["selina zifer, mythology"],
    },
    "kyra_(atrolux)": {
        "character": ["kyra_(atrolux)"],
        "trigger": ["kyra \\(atrolux\\), patreon"],
    },
    "naser_(gvh)": {
        "character": ["naser_(gvh)"],
        "trigger": ["naser \\(gvh\\), goodbye volcano high"],
    },
    "mamoru-kun": {
        "character": ["mamoru-kun"],
        "trigger": ["mamoru-kun, little tail bronx"],
    },
    "lady_nora_(twokinds)": {
        "character": ["lady_nora_(twokinds)"],
        "trigger": ["lady nora \\(twokinds\\), twokinds"],
    },
    "wolfrun": {"character": ["wolfrun"], "trigger": ["wolfrun, pretty cure"]},
    "reg_(made_in_abyss)": {
        "character": ["reg_(made_in_abyss)"],
        "trigger": ["reg \\(made in abyss\\), made in abyss"],
    },
    "dasa": {"character": ["dasa"], "trigger": ["dasa, mythology"]},
    "giygas": {"character": ["giygas"], "trigger": ["giygas, earthbound \\(series\\)"]},
    "sugar_belle_(mlp)": {
        "character": ["sugar_belle_(mlp)"],
        "trigger": ["sugar belle \\(mlp\\), my little pony"],
    },
    "hombre_tigre_(tas)": {
        "character": ["hombre_tigre_(tas)"],
        "trigger": ["hombre tigre \\(tas\\), lifewonders"],
    },
    "von_lycaon": {"character": ["von_lycaon"], "trigger": ["von lycaon, mihoyo"]},
    "ami_bandicoot": {
        "character": ["ami_bandicoot"],
        "trigger": ["ami bandicoot, crash bandicoot \\(series\\)"],
    },
    "rune_(wooled)": {
        "character": ["rune_(wooled)"],
        "trigger": ["rune \\(wooled\\), pokemon"],
    },
    "hoodwink_(dota)": {
        "character": ["hoodwink_(dota)"],
        "trigger": ["hoodwink \\(dota\\), dota"],
    },
    "gordi_(unicorn_wars)": {
        "character": ["gordi_(unicorn_wars)"],
        "trigger": ["gordi \\(unicorn wars\\), unicorn wars"],
    },
    "munks_(munkeesgomu)": {
        "character": ["munks_(munkeesgomu)"],
        "trigger": ["munks \\(munkeesgomu\\), snapchat"],
    },
    "demon_lord_dragon_batzz": {
        "character": ["demon_lord_dragon_batzz"],
        "trigger": ["demon lord dragon batzz, future card buddyfight"],
    },
    "follower_(cult_of_the_lamb)": {
        "character": ["follower_(cult_of_the_lamb)"],
        "trigger": ["follower \\(cult of the lamb\\), cult of the lamb"],
    },
    "leon_kennedy": {
        "character": ["leon_kennedy"],
        "trigger": ["leon kennedy, resident evil"],
    },
    "crunch_bandicoot": {
        "character": ["crunch_bandicoot"],
        "trigger": ["crunch bandicoot, crash bandicoot \\(series\\)"],
    },
    "ganba": {
        "character": ["ganba"],
        "trigger": ["ganba, gamba no bouken \\(series\\)"],
    },
    "padre_(unicorn_wars)": {
        "character": ["padre_(unicorn_wars)"],
        "trigger": ["padre \\(unicorn wars\\), unicorn wars"],
    },
    "kuromi": {"character": ["kuromi"], "trigger": ["kuromi, onegai my melody"]},
    "tayelle_ebonclaw": {
        "character": ["tayelle_ebonclaw"],
        "trigger": ["tayelle ebonclaw, square enix"],
    },
    "kumatetsu": {
        "character": ["kumatetsu"],
        "trigger": ["kumatetsu, the boy and the beast"],
    },
    "yona_yak_(mlp)": {
        "character": ["yona_yak_(mlp)"],
        "trigger": ["yona yak \\(mlp\\), my little pony"],
    },
    "lupe_the_wolf": {
        "character": ["lupe_the_wolf"],
        "trigger": ["lupe the wolf, sonic the hedgehog \\(series\\)"],
    },
    "squigly": {"character": ["squigly"], "trigger": ["squigly, skullgirls"]},
    "graff_filsh": {
        "character": ["graff_filsh"],
        "trigger": ["graff filsh, brok the investigator"],
    },
    "hopey": {"character": ["hopey"], "trigger": ["hopey, mythology"]},
    "garmr_(tas)": {
        "character": ["garmr_(tas)"],
        "trigger": ["garmr \\(tas\\), lifewonders"],
    },
    "jimmy_crystal": {
        "character": ["jimmy_crystal"],
        "trigger": ["jimmy crystal, illumination entertainment"],
    },
    "siroc_(character)": {
        "character": ["siroc_(character)"],
        "trigger": ["siroc \\(character\\), disney"],
    },
    "jay_(1-upclock)": {
        "character": ["jay_(1-upclock)"],
        "trigger": ["jay \\(1-upclock\\), nintendo"],
    },
    "sebastien_(black-kitten)": {
        "character": ["sebastien_(black-kitten)"],
        "trigger": ["sebastien \\(black-kitten\\), christmas"],
    },
    "syrazor": {"character": ["syrazor"], "trigger": ["syrazor, mythology"]},
    "gaster": {"character": ["gaster"], "trigger": ["gaster, undertale \\(series\\)"]},
    "viola_bat_(character)": {
        "character": ["viola_bat_(character)"],
        "trigger": ["viola bat \\(character\\), mythology"],
    },
    "gyumao_(tas)": {
        "character": ["gyumao_(tas)"],
        "trigger": ["gyumao \\(tas\\), lifewonders"],
    },
    "grape_jelly_(housepets!)": {
        "character": ["grape_jelly_(housepets!)"],
        "trigger": ["grape jelly \\(housepets!\\), housepets!"],
    },
    "brittany_miller": {
        "character": ["brittany_miller"],
        "trigger": ["brittany miller, alvin and the chipmunks"],
    },
    "tracer_(overwatch)": {
        "character": ["tracer_(overwatch)"],
        "trigger": ["tracer \\(overwatch\\), overwatch"],
    },
    "gyro_tech": {"character": ["gyro_tech"], "trigger": ["gyro tech, mythology"]},
    "chloe_sinclaire": {
        "character": ["chloe_sinclaire"],
        "trigger": ["chloe sinclaire, christmas"],
    },
    "sythe_(twokinds)": {
        "character": ["sythe_(twokinds)"],
        "trigger": ["sythe \\(twokinds\\), twokinds"],
    },
    "jennifer_(study_partners)": {
        "character": ["jennifer_(study_partners)"],
        "trigger": ["jennifer \\(study partners\\), study partners"],
    },
    "shoen": {"character": ["shoen"], "trigger": ["shoen, lifewonders"]},
    "demona_(gargoyles)": {
        "character": ["demona_(gargoyles)"],
        "trigger": ["demona \\(gargoyles\\), disney"],
    },
    "nameless_character": {
        "character": ["nameless_character"],
        "trigger": ["nameless character, mythology"],
    },
    "felina_feral": {
        "character": ["felina_feral"],
        "trigger": ["felina feral, swat kats"],
    },
    "hilbert_(pokemon)": {
        "character": ["hilbert_(pokemon)"],
        "trigger": ["hilbert \\(pokemon\\), pokemon"],
    },
    "cloudy_quartz_(mlp)": {
        "character": ["cloudy_quartz_(mlp)"],
        "trigger": ["cloudy quartz \\(mlp\\), my little pony"],
    },
    "bratty_(undertale)": {
        "character": ["bratty_(undertale)"],
        "trigger": ["bratty \\(undertale\\), undertale \\(series\\)"],
    },
    "shadowbolts_(mlp)": {
        "character": ["shadowbolts_(mlp)"],
        "trigger": ["shadowbolts \\(mlp\\), my little pony"],
    },
    "bianca_(spyro)": {
        "character": ["bianca_(spyro)"],
        "trigger": ["bianca \\(spyro\\), spyro the dragon"],
    },
    "ulti_(ultilix)": {
        "character": ["ulti_(ultilix)"],
        "trigger": ["ulti \\(ultilix\\), nintendo"],
    },
    "female_protagonist_(tas)": {
        "character": ["female_protagonist_(tas)"],
        "trigger": ["female protagonist \\(tas\\), lifewonders"],
    },
    "mr._peanutbutter": {
        "character": ["mr._peanutbutter"],
        "trigger": ["mr. peanutbutter, netflix"],
    },
    "shota_deer_(berseepon09)": {
        "character": ["shota_deer_(berseepon09)"],
        "trigger": ["shota deer \\(berseepon09\\), mythology"],
    },
    "percy_(teckworks)": {
        "character": ["percy_(teckworks)"],
        "trigger": ["percy \\(teckworks\\), nintendo"],
    },
    "pepper_(paladins)": {
        "character": ["pepper_(paladins)"],
        "trigger": ["pepper \\(paladins\\), paladins \\(game\\)"],
    },
    "mis'alia": {"character": ["mis'alia"], "trigger": ["mis'alia, mythology"]},
    "artie": {"character": ["artie"], "trigger": ["artie, nintendo"]},
    "spy_(team_fortress_2)": {
        "character": ["spy_(team_fortress_2)"],
        "trigger": ["spy \\(team fortress 2\\), team fortress 2"],
    },
    "kyera": {"character": ["kyera"], "trigger": ["kyera, mythology"]},
    "power_ponies_(mlp)": {
        "character": ["power_ponies_(mlp)"],
        "trigger": ["power ponies \\(mlp\\), my little pony"],
    },
    "king_(tekken)": {
        "character": ["king_(tekken)"],
        "trigger": ["king \\(tekken\\), tekken"],
    },
    "teri_(tawog)": {
        "character": ["teri_(tawog)"],
        "trigger": ["teri \\(tawog\\), cartoon network"],
    },
    "sleepy_(sleepylp)": {
        "character": ["sleepy_(sleepylp)"],
        "trigger": ["sleepy \\(sleepylp\\), disney"],
    },
    "emerald_jewel_(colt_quest)": {
        "character": ["emerald_jewel_(colt_quest)"],
        "trigger": ["emerald jewel \\(colt quest\\), my little pony"],
    },
    "nomad_(tas)": {
        "character": ["nomad_(tas)"],
        "trigger": ["nomad \\(tas\\), lifewonders"],
    },
    "june_(jinu)": {
        "character": ["june_(jinu)"],
        "trigger": ["june \\(jinu\\), nintendo"],
    },
    "curly_brace": {
        "character": ["curly_brace"],
        "trigger": ["curly brace, cave story"],
    },
    "zac_(lol)": {"character": ["zac_(lol)"], "trigger": ["zac \\(lol\\), riot games"]},
    "alexstrasza": {"character": ["alexstrasza"], "trigger": ["alexstrasza, warcraft"]},
    "alt": {"character": ["alt"], "trigger": ["alt, warcraft"]},
    "lam-chan": {"character": ["lam-chan"], "trigger": ["lam-chan, christmas"]},
    "spring_bonnie_(fnaf)": {
        "character": ["spring_bonnie_(fnaf)"],
        "trigger": ["spring bonnie \\(fnaf\\), scottgames"],
    },
    "gin_(twitchyanimation)": {
        "character": ["gin_(twitchyanimation)"],
        "trigger": ["gin \\(twitchyanimation\\), source filmmaker"],
    },
    "midori_(nakagami_takashi)": {
        "character": ["midori_(nakagami_takashi)"],
        "trigger": ["midori \\(nakagami takashi\\), kemokko lovers"],
    },
    "trevor-fox_(character)": {
        "character": ["trevor-fox_(character)"],
        "trigger": ["trevor-fox \\(character\\), nintendo"],
    },
    "kimun_kamui_(tas)": {
        "character": ["kimun_kamui_(tas)"],
        "trigger": ["kimun kamui \\(tas\\), lifewonders"],
    },
    "liz_bandicoot": {
        "character": ["liz_bandicoot"],
        "trigger": ["liz bandicoot, crash bandicoot \\(series\\)"],
    },
    "qhala": {"character": ["qhala"], "trigger": ["qhala, source filmmaker"]},
    "sphinx_(mlp)": {
        "character": ["sphinx_(mlp)"],
        "trigger": ["sphinx \\(mlp\\), my little pony"],
    },
    "jade_harley": {
        "character": ["jade_harley"],
        "trigger": ["jade harley, homestuck"],
    },
    "cerberus_(helltaker)": {
        "character": ["cerberus_(helltaker)"],
        "trigger": ["cerberus \\(helltaker\\), helltaker"],
    },
    "wolflong_(character)": {
        "character": ["wolflong_(character)"],
        "trigger": ["wolflong \\(character\\), mythology"],
    },
    "pina_(beastars)": {
        "character": ["pina_(beastars)"],
        "trigger": ["pina \\(beastars\\), beastars"],
    },
    "cleo_catillac": {
        "character": ["cleo_catillac"],
        "trigger": ["cleo catillac, heathcliff and the catillac cats"],
    },
    "stripes_(character)": {
        "character": ["stripes_(character)"],
        "trigger": ["stripes \\(character\\), mythology"],
    },
    "larry_koopa": {
        "character": ["larry_koopa"],
        "trigger": ["larry koopa, mario bros"],
    },
    "hetty_(faf)": {
        "character": ["hetty_(faf)"],
        "trigger": ["hetty \\(faf\\), fafcomics"],
    },
    "tiffy_(fastrunner2024)": {
        "character": ["tiffy_(fastrunner2024)"],
        "trigger": ["tiffy \\(fastrunner2024\\), christmas"],
    },
    "lancer_(deltarune)": {
        "character": ["lancer_(deltarune)"],
        "trigger": ["lancer \\(deltarune\\), undertale \\(series\\)"],
    },
    "firebrand": {"character": ["firebrand"], "trigger": ["firebrand, demon's crest"]},
    "madelyn_adelaide": {
        "character": ["madelyn_adelaide"],
        "trigger": ["madelyn adelaide, twokinds"],
    },
    "navarchus_zepto": {
        "character": ["navarchus_zepto"],
        "trigger": ["navarchus zepto, mythology"],
    },
    "delilah_(101_dalmatians)": {
        "character": ["delilah_(101_dalmatians)"],
        "trigger": ["delilah \\(101 dalmatians\\), disney"],
    },
    "tobi_(nimzy)": {
        "character": ["tobi_(nimzy)"],
        "trigger": ["tobi \\(nimzy\\), christmas"],
    },
    "mineru": {"character": ["mineru"], "trigger": ["mineru, the legend of zelda"]},
    "m'ress": {
        "character": ["m'ress"],
        "trigger": ["m'ress, star trek the animated series"],
    },
    "lisa_simpson": {
        "character": ["lisa_simpson"],
        "trigger": ["lisa simpson, the simpsons"],
    },
    "iggy_(jjba)": {
        "character": ["iggy_(jjba)"],
        "trigger": ["iggy \\(jjba\\), jojo's bizarre adventure"],
    },
    "dewey_duck": {"character": ["dewey_duck"], "trigger": ["dewey duck, disney"]},
    "chari-gal": {"character": ["chari-gal"], "trigger": ["chari-gal, pokemon"]},
    "chernobog_(tas)": {
        "character": ["chernobog_(tas)"],
        "trigger": ["chernobog \\(tas\\), lifewonders"],
    },
    "mugman": {"character": ["mugman"], "trigger": ["mugman, cuphead \\(game\\)"]},
    "sheila_vixen": {
        "character": ["sheila_vixen"],
        "trigger": ["sheila vixen, furafterdark"],
    },
    "dean_(drako1997)": {
        "character": ["dean_(drako1997)"],
        "trigger": ["dean \\(drako1997\\), nintendo"],
    },
    "heavy_(team_fortress_2)": {
        "character": ["heavy_(team_fortress_2)"],
        "trigger": ["heavy \\(team fortress 2\\), valve"],
    },
    "nicobay": {"character": ["nicobay"], "trigger": ["nicobay, pokemon"]},
    "tai_lung_(kung_fu_panda)": {
        "character": ["tai_lung_(kung_fu_panda)"],
        "trigger": ["tai lung \\(kung fu panda\\), kung fu panda"],
    },
    "luskfoxx": {"character": ["luskfoxx"], "trigger": ["luskfoxx, mythology"]},
    "kassen_akoll": {
        "character": ["kassen_akoll"],
        "trigger": ["kassen akoll, out-of-placers"],
    },
    "serval-chan": {
        "character": ["serval-chan"],
        "trigger": ["serval-chan, kemono friends"],
    },
    "jevil_(deltarune)": {
        "character": ["jevil_(deltarune)"],
        "trigger": ["jevil \\(deltarune\\), undertale \\(series\\)"],
    },
    "peanut_butter_(housepets!)": {
        "character": ["peanut_butter_(housepets!)"],
        "trigger": ["peanut butter \\(housepets!\\), housepets!"],
    },
    "ronno": {"character": ["ronno"], "trigger": ["ronno, disney"]},
    "curt_(animal_crossing)": {
        "character": ["curt_(animal_crossing)"],
        "trigger": ["curt \\(animal crossing\\), animal crossing"],
    },
    "lewdtias": {"character": ["lewdtias"], "trigger": ["lewdtias, pokemon"]},
    "navi": {"character": ["navi"], "trigger": ["navi, the legend of zelda"]},
    "fang_the_hunter": {
        "character": ["fang_the_hunter"],
        "trigger": ["fang the hunter, sonic the hedgehog \\(series\\)"],
    },
    "spot_(arknights)": {
        "character": ["spot_(arknights)"],
        "trigger": ["spot \\(arknights\\), studio montagne"],
    },
    "huey_duck": {"character": ["huey_duck"], "trigger": ["huey duck, disney"]},
    "sajin_komamura": {
        "character": ["sajin_komamura"],
        "trigger": ["sajin komamura, bleach \\(series\\)"],
    },
    "jinbe": {"character": ["jinbe"], "trigger": ["jinbe, one piece"]},
    "reggie_(whygena)": {
        "character": ["reggie_(whygena)"],
        "trigger": ["reggie \\(whygena\\), snapchat"],
    },
    "ara_(fluff-kevlar)": {
        "character": ["ara_(fluff-kevlar)"],
        "trigger": ["ara \\(fluff-kevlar\\), christmas"],
    },
    "drone_(mlp)": {
        "character": ["drone_(mlp)"],
        "trigger": ["drone \\(mlp\\), my little pony"],
    },
    "behemoth_(tas)": {
        "character": ["behemoth_(tas)"],
        "trigger": ["behemoth \\(tas\\), lifewonders"],
    },
    "pit_(kid_icarus)": {
        "character": ["pit_(kid_icarus)"],
        "trigger": ["pit \\(kid icarus\\), kid icarus"],
    },
    "grunt_(pokemon)": {
        "character": ["grunt_(pokemon)"],
        "trigger": ["grunt \\(pokemon\\), pokemon"],
    },
    "katt_(animal_crossing)": {
        "character": ["katt_(animal_crossing)"],
        "trigger": ["katt \\(animal crossing\\), animal crossing"],
    },
    "temujin_(tas)": {
        "character": ["temujin_(tas)"],
        "trigger": ["temujin \\(tas\\), lifewonders"],
    },
    "law_(sdorica)": {
        "character": ["law_(sdorica)"],
        "trigger": ["law \\(sdorica\\), sdorica"],
    },
    "ben_tennyson": {
        "character": ["ben_tennyson"],
        "trigger": ["ben tennyson, cartoon network"],
    },
    "viriathus_vayu": {
        "character": ["viriathus_vayu"],
        "trigger": ["viriathus vayu, dreamkeepers"],
    },
    "king_boo": {"character": ["king_boo"], "trigger": ["king boo, mario bros"]},
    "tianhuo_(tfh)": {
        "character": ["tianhuo_(tfh)"],
        "trigger": ["tianhuo \\(tfh\\), them's fightin' herds"],
    },
    "alphonse_(james_howard)": {
        "character": ["alphonse_(james_howard)"],
        "trigger": ["alphonse \\(james howard\\), patreon"],
    },
    "fara_phoenix": {
        "character": ["fara_phoenix"],
        "trigger": ["fara phoenix, star fox"],
    },
    "my_melody": {
        "character": ["my_melody"],
        "trigger": ["my melody, onegai my melody"],
    },
    "counting_cougar": {
        "character": ["counting_cougar"],
        "trigger": ["counting cougar, t.u.f.f. puppy"],
    },
    "quetzalli_(character)": {
        "character": ["quetzalli_(character)"],
        "trigger": ["quetzalli \\(character\\), pokemon"],
    },
    "pom_(tfh)": {
        "character": ["pom_(tfh)"],
        "trigger": ["pom \\(tfh\\), them's fightin' herds"],
    },
    "bingo_heeler": {
        "character": ["bingo_heeler"],
        "trigger": ["bingo heeler, bluey \\(series\\)"],
    },
    "marie_itami": {
        "character": ["marie_itami"],
        "trigger": ["marie itami, studio trigger"],
    },
    "fizzarolli_(helluva_boss)": {
        "character": ["fizzarolli_(helluva_boss)"],
        "trigger": ["fizzarolli \\(helluva boss\\), helluva boss"],
    },
    "ivy_pepper": {"character": ["ivy_pepper"], "trigger": ["ivy pepper, lackadaisy"]},
    "fleetfoot_(mlp)": {
        "character": ["fleetfoot_(mlp)"],
        "trigger": ["fleetfoot \\(mlp\\), my little pony"],
    },
    "napstablook": {
        "character": ["napstablook"],
        "trigger": ["napstablook, undertale \\(series\\)"],
    },
    "senky": {"character": ["senky"], "trigger": ["senky, mythology"]},
    "eti_(utopianvee)": {
        "character": ["eti_(utopianvee)"],
        "trigger": ["eti \\(utopianvee\\), pokemon"],
    },
    "dripdry": {"character": ["dripdry"], "trigger": ["dripdry, mythology"]},
    "namah_calah": {
        "character": ["namah_calah"],
        "trigger": ["namah calah, dreamkeepers"],
    },
    "caramel_(mlp)": {
        "character": ["caramel_(mlp)"],
        "trigger": ["caramel \\(mlp\\), my little pony"],
    },
    "zoran": {"character": ["zoran"], "trigger": ["zoran, mythology"]},
    "woody_(study_partners)": {
        "character": ["woody_(study_partners)"],
        "trigger": ["woody \\(study partners\\), study partners"],
    },
    "shadow_(lol)": {
        "character": ["shadow_(lol)"],
        "trigger": ["shadow \\(lol\\), riot games"],
    },
    "shun_(morenatsu)": {
        "character": ["shun_(morenatsu)"],
        "trigger": ["shun \\(morenatsu\\), morenatsu"],
    },
    "mammon_(helluva_boss)": {
        "character": ["mammon_(helluva_boss)"],
        "trigger": ["mammon \\(helluva boss\\), helluva boss"],
    },
    "aryanne_(character)": {
        "character": ["aryanne_(character)"],
        "trigger": ["aryanne \\(character\\), my little pony"],
    },
    "runaboo_chica": {
        "character": ["runaboo_chica"],
        "trigger": ["runaboo chica, five nights at freddy's 2"],
    },
    "victor_(pokemon)": {
        "character": ["victor_(pokemon)"],
        "trigger": ["victor \\(pokemon\\), pokemon"],
    },
    "iscream_(chikn_nuggit)": {
        "character": ["iscream_(chikn_nuggit)"],
        "trigger": ["iscream \\(chikn nuggit\\), chikn nuggit"],
    },
    "zhurong_(tas)": {
        "character": ["zhurong_(tas)"],
        "trigger": ["zhurong \\(tas\\), lifewonders"],
    },
    "link_(rabbit_form)": {
        "character": ["link_(rabbit_form)"],
        "trigger": ["link \\(rabbit form\\), the legend of zelda"],
    },
    "cloves_(freckles)": {
        "character": ["cloves_(freckles)"],
        "trigger": ["cloves \\(freckles\\), mythology"],
    },
    "nia_(senz)": {"character": ["nia_(senz)"], "trigger": ["nia \\(senz\\), patreon"]},
    "roi_(smuttysquid)": {
        "character": ["roi_(smuttysquid)"],
        "trigger": ["roi \\(smuttysquid\\), digimon"],
    },
    "katy_kat": {
        "character": ["katy_kat"],
        "trigger": ["katy kat, parappa the rapper"],
    },
    "chaos_(sonic)": {
        "character": ["chaos_(sonic)"],
        "trigger": ["chaos \\(sonic\\), sonic the hedgehog \\(series\\)"],
    },
    "candy_kong": {
        "character": ["candy_kong"],
        "trigger": ["candy kong, donkey kong \\(series\\)"],
    },
    "der": {"character": ["der"], "trigger": ["der, mythology"]},
    "twist_(mlp)": {
        "character": ["twist_(mlp)"],
        "trigger": ["twist \\(mlp\\), my little pony"],
    },
    "geeflakes_(character)": {
        "character": ["geeflakes_(character)"],
        "trigger": ["geeflakes \\(character\\), nintendo"],
    },
    "hazel_(animal_crossing)": {
        "character": ["hazel_(animal_crossing)"],
        "trigger": ["hazel \\(animal crossing\\), animal crossing"],
    },
    "arsalan_(tas)": {
        "character": ["arsalan_(tas)"],
        "trigger": ["arsalan \\(tas\\), lifewonders"],
    },
    "cookie_(furryfight_chronicles)": {
        "character": ["cookie_(furryfight_chronicles)"],
        "trigger": ["cookie \\(furryfight chronicles\\), furryfight chronicles"],
    },
    "infinite": {"character": ["infinite"], "trigger": ["infinite, mythology"]},
    "delga": {"character": ["delga"], "trigger": ["delga, mythology"]},
    "neko_ed": {"character": ["neko_ed"], "trigger": ["neko ed, mythology"]},
    "mia_mouse": {"character": ["mia_mouse"], "trigger": ["mia mouse, halloween"]},
    "kirara_(inuyasha)": {
        "character": ["kirara_(inuyasha)"],
        "trigger": ["kirara \\(inuyasha\\), inuyasha"],
    },
    "crocodine": {"character": ["crocodine"], "trigger": ["crocodine, square enix"]},
    "clovis_(twokinds)": {
        "character": ["clovis_(twokinds)"],
        "trigger": ["clovis \\(twokinds\\), twokinds"],
    },
    "mrs._otterton": {
        "character": ["mrs._otterton"],
        "trigger": ["mrs. otterton, disney"],
    },
    "blahaj": {"character": ["blahaj"], "trigger": ["blahaj, ikea"]},
    "kipfox": {"character": ["kipfox"], "trigger": ["kipfox, mythology"]},
    "annoying_dog_(undertale)": {
        "character": ["annoying_dog_(undertale)"],
        "trigger": ["annoying dog \\(undertale\\), undertale \\(series\\)"],
    },
    "teddy_(animal_crossing)": {
        "character": ["teddy_(animal_crossing)"],
        "trigger": ["teddy \\(animal crossing\\), animal crossing"],
    },
    "sargento_caricias": {
        "character": ["sargento_caricias"],
        "trigger": ["sargento caricias, unicorn wars"],
    },
    "suirano_(character)": {
        "character": ["suirano_(character)"],
        "trigger": ["suirano \\(character\\), mythology"],
    },
    "meagan_(silver_soul)": {
        "character": ["meagan_(silver_soul)"],
        "trigger": ["meagan \\(silver soul\\), pokemon"],
    },
    "inui_(aggretsuko)": {
        "character": ["inui_(aggretsuko)"],
        "trigger": ["inui \\(aggretsuko\\), sanrio"],
    },
    "striker_(helluva_boss)": {
        "character": ["striker_(helluva_boss)"],
        "trigger": ["striker \\(helluva boss\\), helluva boss"],
    },
    "ashchu": {"character": ["ashchu"], "trigger": ["ashchu, pokemon"]},
    "dragon_(shrek)": {
        "character": ["dragon_(shrek)"],
        "trigger": ["dragon \\(shrek\\), shrek \\(series\\)"],
    },
    "margaret_smith_(regular_show)": {
        "character": ["margaret_smith_(regular_show)"],
        "trigger": ["margaret smith \\(regular show\\), cartoon network"],
    },
    "lir_(icma)": {"character": ["lir_(icma)"], "trigger": ["lir \\(icma\\), pokemon"]},
    "pete_(disney)": {
        "character": ["pete_(disney)"],
        "trigger": ["pete \\(disney\\), disney"],
    },
    "garble_(mlp)": {
        "character": ["garble_(mlp)"],
        "trigger": ["garble \\(mlp\\), my little pony"],
    },
    "yooka": {"character": ["yooka"], "trigger": ["yooka, playtonic games"]},
    "husk_(hazbin_hotel)": {
        "character": ["husk_(hazbin_hotel)"],
        "trigger": ["husk \\(hazbin hotel\\), hazbin hotel"],
    },
    "bahamut": {"character": ["bahamut"], "trigger": ["bahamut, mythology"]},
    "haley_(nightfaux)": {
        "character": ["haley_(nightfaux)"],
        "trigger": ["haley \\(nightfaux\\), digimon"],
    },
    "sarah_(study_partners)": {
        "character": ["sarah_(study_partners)"],
        "trigger": ["sarah \\(study partners\\), study partners"],
    },
    "mashiro": {"character": ["mashiro"], "trigger": ["mashiro, april fools' day"]},
    "gadget_the_wolf": {
        "character": ["gadget_the_wolf"],
        "trigger": ["gadget the wolf, sonic the hedgehog \\(series\\)"],
    },
    "ophion_(tas)": {
        "character": ["ophion_(tas)"],
        "trigger": ["ophion \\(tas\\), lifewonders"],
    },
    "farah_(legend_of_queen_opala)": {
        "character": ["farah_(legend_of_queen_opala)"],
        "trigger": ["farah \\(legend of queen opala\\), legend of queen opala"],
    },
    "namihira_kousuke": {
        "character": ["namihira_kousuke"],
        "trigger": ["namihira kousuke, trouble \\(series\\)"],
    },
    "naomi_rasputin": {
        "character": ["naomi_rasputin"],
        "trigger": ["naomi rasputin, pokemon"],
    },
    "lolly_(animal_crossing)": {
        "character": ["lolly_(animal_crossing)"],
        "trigger": ["lolly \\(animal crossing\\), animal crossing"],
    },
    "asmodeus_(helluva_boss)": {
        "character": ["asmodeus_(helluva_boss)"],
        "trigger": ["asmodeus \\(helluva boss\\), helluva boss"],
    },
    "misty_brightdawn_(mlp)": {
        "character": ["misty_brightdawn_(mlp)"],
        "trigger": ["misty brightdawn \\(mlp\\), my little pony"],
    },
    "chun-li": {"character": ["chun-li"], "trigger": ["chun-li, street fighter"]},
    "mabel_pines": {"character": ["mabel_pines"], "trigger": ["mabel pines, disney"]},
    "storm_(stormwx_wolf)": {
        "character": ["storm_(stormwx_wolf)"],
        "trigger": ["storm \\(stormwx wolf\\), mythology"],
    },
    "funtime_freddy_(fnafsl)": {
        "character": ["funtime_freddy_(fnafsl)"],
        "trigger": ["funtime freddy \\(fnafsl\\), scottgames"],
    },
    "pubraseer": {"character": ["pubraseer"], "trigger": ["pubraseer, lifewonders"]},
    "naruto_uzumaki": {
        "character": ["naruto_uzumaki"],
        "trigger": ["naruto uzumaki, naruto"],
    },
    "label_able": {
        "character": ["label_able"],
        "trigger": ["label able, animal crossing"],
    },
    "coloratura_(mlp)": {
        "character": ["coloratura_(mlp)"],
        "trigger": ["coloratura \\(mlp\\), my little pony"],
    },
    "mary_senicourt": {
        "character": ["mary_senicourt"],
        "trigger": ["mary senicourt, cartoon network"],
    },
    "danny_sterling_(spitfire420007)": {
        "character": ["danny_sterling_(spitfire420007)"],
        "trigger": ["danny sterling \\(spitfire420007\\), blender \\(software\\)"],
    },
    "midna_(true_form)": {
        "character": ["midna_(true_form)"],
        "trigger": ["midna \\(true form\\), twilight princess"],
    },
    "spartan_(halo)": {
        "character": ["spartan_(halo)"],
        "trigger": ["spartan \\(halo\\), halo \\(series\\)"],
    },
    "doggo_(undertale)": {
        "character": ["doggo_(undertale)"],
        "trigger": ["doggo \\(undertale\\), undertale \\(series\\)"],
    },
    "zerva_von_zadok_(capesir)": {
        "character": ["zerva_von_zadok_(capesir)"],
        "trigger": ["zerva von zadok \\(capesir\\), mythology"],
    },
    "general_scales": {
        "character": ["general_scales"],
        "trigger": ["general scales, star fox"],
    },
    "cofi_(chikn_nuggit)": {
        "character": ["cofi_(chikn_nuggit)"],
        "trigger": ["cofi \\(chikn nuggit\\), chikn nuggit"],
    },
    "sonia_the_hedgehog": {
        "character": ["sonia_the_hedgehog"],
        "trigger": ["sonia the hedgehog, sonic the hedgehog \\(series\\)"],
    },
    "master_crane": {
        "character": ["master_crane"],
        "trigger": ["master crane, kung fu panda"],
    },
    "jeane_(ceehaz)": {
        "character": ["jeane_(ceehaz)"],
        "trigger": ["jeane \\(ceehaz\\), dog knight rpg"],
    },
    "alcina_dimitrescu": {
        "character": ["alcina_dimitrescu"],
        "trigger": ["alcina dimitrescu, resident evil"],
    },
    "mr._piranha_(the_bad_guys)": {
        "character": ["mr._piranha_(the_bad_guys)"],
        "trigger": ["mr. piranha \\(the bad guys\\), the bad guys"],
    },
    "nytro_(fluff-kevlar)": {
        "character": ["nytro_(fluff-kevlar)"],
        "trigger": ["nytro \\(fluff-kevlar\\), university tails"],
    },
    "mace_(dreamkeepers)": {
        "character": ["mace_(dreamkeepers)"],
        "trigger": ["mace \\(dreamkeepers\\), dreamkeepers"],
    },
    "kuroodod_(character)": {
        "character": ["kuroodod_(character)"],
        "trigger": ["kuroodod \\(character\\), pokemon"],
    },
    "mrs._nibbly": {"character": ["mrs._nibbly"], "trigger": ["mrs. nibbly, twokinds"]},
    "cirrus_(xp)": {
        "character": ["cirrus_(xp)"],
        "trigger": ["cirrus \\(xp\\), mythology"],
    },
    "rakan": {"character": ["rakan"], "trigger": ["rakan, mythology"]},
    "dreamy_pride_(character)": {
        "character": ["dreamy_pride_(character)"],
        "trigger": ["dreamy pride \\(character\\), mythology"],
    },
    "dylan_(zourik)": {
        "character": ["dylan_(zourik)"],
        "trigger": ["dylan \\(zourik\\), pokemon"],
    },
    "bubsy": {"character": ["bubsy"], "trigger": ["bubsy, bubsy \\(series\\)"]},
    "pyravia": {"character": ["pyravia"], "trigger": ["pyravia, mythology"]},
    "senior_fox": {"character": ["senior_fox"], "trigger": ["senior fox, swat"]},
    "ragdoll_(study_partners)": {
        "character": ["ragdoll_(study_partners)"],
        "trigger": ["ragdoll \\(study partners\\), study partners"],
    },
    "sarah_silkie": {
        "character": ["sarah_silkie"],
        "trigger": ["sarah silkie, las lindas"],
    },
    "andrew_oleander": {
        "character": ["andrew_oleander"],
        "trigger": ["andrew oleander, summers gone"],
    },
    "kaz_(kazudanefonfon)": {
        "character": ["kaz_(kazudanefonfon)"],
        "trigger": ["kaz \\(kazudanefonfon\\), mythology"],
    },
    "jeanette_miller": {
        "character": ["jeanette_miller"],
        "trigger": ["jeanette miller, alvin and the chipmunks"],
    },
    "bendy_the_dancing_demon": {
        "character": ["bendy_the_dancing_demon"],
        "trigger": ["bendy the dancing demon, bendy and the ink machine"],
    },
    "shinobu": {"character": ["shinobu"], "trigger": ["shinobu, xbox game studios"]},
    "cteno": {"character": ["cteno"], "trigger": ["cteno, ctenophorae"]},
    "toy_freddy_(psychojohn2)": {
        "character": ["toy_freddy_(psychojohn2)"],
        "trigger": ["toy freddy \\(psychojohn2\\), scottgames"],
    },
    "roz_(erobos)": {
        "character": ["roz_(erobos)"],
        "trigger": ["roz \\(erobos\\), cartoon network"],
    },
    "sir_gallade": {"character": ["sir_gallade"], "trigger": ["sir gallade, pokemon"]},
    "azulin_(unicorn_wars)": {
        "character": ["azulin_(unicorn_wars)"],
        "trigger": ["azulin \\(unicorn wars\\), unicorn wars"],
    },
    "brendan_(pokemon)": {
        "character": ["brendan_(pokemon)"],
        "trigger": ["brendan \\(pokemon\\), pokemon"],
    },
    "kagerou_imaizumi": {
        "character": ["kagerou_imaizumi"],
        "trigger": ["kagerou imaizumi, touhou"],
    },
    "jambavan_(tas)": {
        "character": ["jambavan_(tas)"],
        "trigger": ["jambavan \\(tas\\), lifewonders"],
    },
    "fukiyo_(pokeandpenetrate)": {
        "character": ["fukiyo_(pokeandpenetrate)"],
        "trigger": ["fukiyo \\(pokeandpenetrate\\), square enix"],
    },
    "riz_(beastars)": {
        "character": ["riz_(beastars)"],
        "trigger": ["riz \\(beastars\\), beastars"],
    },
    "zoologist_(terraria)": {
        "character": ["zoologist_(terraria)"],
        "trigger": ["zoologist \\(terraria\\), terraria"],
    },
    "patrick_(kadath)": {
        "character": ["patrick_(kadath)"],
        "trigger": ["patrick \\(kadath\\), patreon"],
    },
    "june_(justathereptile)": {
        "character": ["june_(justathereptile)"],
        "trigger": ["june \\(justathereptile\\), sonic the hedgehog \\(series\\)"],
    },
    "synx_(synxthelynx)": {
        "character": ["synx_(synxthelynx)"],
        "trigger": ["synx \\(synxthelynx\\), mythology"],
    },
    "yohack": {"character": ["yohack"], "trigger": ["yohack, lifewonders"]},
    "katz_(courage_the_cowardly_dog)": {
        "character": ["katz_(courage_the_cowardly_dog)"],
        "trigger": ["katz \\(courage the cowardly dog\\), cartoon network"],
    },
    "ziggs_(lol)": {
        "character": ["ziggs_(lol)"],
        "trigger": ["ziggs \\(lol\\), riot games"],
    },
    "mutt_(wagnermutt)": {
        "character": ["mutt_(wagnermutt)"],
        "trigger": ["mutt \\(wagnermutt\\), nintendo"],
    },
    "cooper_estevez": {
        "character": ["cooper_estevez"],
        "trigger": ["cooper estevez, summers gone"],
    },
    "neri_(caelum_sky)": {
        "character": ["neri_(caelum_sky)"],
        "trigger": ["neri \\(caelum sky\\), caelum sky"],
    },
    "mothra": {"character": ["mothra"], "trigger": ["mothra, mothra \\(series\\)"]},
    "clementine_(aswake)": {
        "character": ["clementine_(aswake)"],
        "trigger": ["clementine \\(aswake\\), a story with a known end"],
    },
    "seph_(naughtymorg)": {
        "character": ["seph_(naughtymorg)"],
        "trigger": ["seph \\(naughtymorg\\), mythology"],
    },
    "medusa": {"character": ["medusa"], "trigger": ["medusa, european mythology"]},
    "goombella": {"character": ["goombella"], "trigger": ["goombella, mario bros"]},
    "futaba_kotobuki": {
        "character": ["futaba_kotobuki"],
        "trigger": ["futaba kotobuki, trouble \\(series\\)"],
    },
    "falla_(f-r95)": {
        "character": ["falla_(f-r95)"],
        "trigger": ["falla \\(f-r95\\), mythology"],
    },
    "detective_pikachu": {
        "character": ["detective_pikachu"],
        "trigger": ["detective pikachu, pokemon"],
    },
    "reisen_udongein_inaba": {
        "character": ["reisen_udongein_inaba"],
        "trigger": ["reisen udongein inaba, touhou"],
    },
    "leia_organa": {
        "character": ["leia_organa"],
        "trigger": ["leia organa, star wars"],
    },
    "dax_(daxzor)": {
        "character": ["dax_(daxzor)"],
        "trigger": ["dax \\(daxzor\\), mythology"],
    },
    "dimmi_(character)": {
        "character": ["dimmi_(character)"],
        "trigger": ["dimmi \\(character\\), mythology"],
    },
    "marco_(angstrom)": {
        "character": ["marco_(angstrom)"],
        "trigger": ["marco \\(angstrom\\), pokemon"],
    },
    "fran_(final_fantasy)": {
        "character": ["fran_(final_fantasy)"],
        "trigger": ["fran \\(final fantasy\\), final fantasy xii"],
    },
    "calvin_mcmurray": {
        "character": ["calvin_mcmurray"],
        "trigger": ["calvin mcmurray, lackadaisy"],
    },
    "lisa_(study_partners)": {
        "character": ["lisa_(study_partners)"],
        "trigger": ["lisa \\(study partners\\), study partners"],
    },
    "bincu": {"character": ["bincu"], "trigger": ["bincu, christmas"]},
    "shooty_(shootysylveon)": {
        "character": ["shooty_(shootysylveon)"],
        "trigger": ["shooty \\(shootysylveon\\), pokemon"],
    },
    "kitsunami_the_fennec": {
        "character": ["kitsunami_the_fennec"],
        "trigger": ["kitsunami the fennec, sonic the hedgehog \\(series\\)"],
    },
    "cobalt_(tatsuchan18)": {
        "character": ["cobalt_(tatsuchan18)"],
        "trigger": ["cobalt \\(tatsuchan18\\), mythology"],
    },
    "ganon": {"character": ["ganon"], "trigger": ["ganon, the legend of zelda"]},
    "lupine_assassin": {
        "character": ["lupine_assassin"],
        "trigger": ["lupine assassin, disney"],
    },
    "rachael_saleigh": {
        "character": ["rachael_saleigh"],
        "trigger": ["rachael saleigh, las lindas"],
    },
    "jake_cottontail": {
        "character": ["jake_cottontail"],
        "trigger": ["jake cottontail, christmas"],
    },
    "isabella_bandicoot": {
        "character": ["isabella_bandicoot"],
        "trigger": ["isabella bandicoot, crash bandicoot \\(series\\)"],
    },
    "palutena": {"character": ["palutena"], "trigger": ["palutena, nintendo"]},
    "sulfer": {"character": ["sulfer"], "trigger": ["sulfer, mythology"]},
    "desi": {
        "character": ["desi"],
        "trigger": ["desi, sonic the hedgehog \\(series\\)"],
    },
    "raricow_(mlp)": {
        "character": ["raricow_(mlp)"],
        "trigger": ["raricow \\(mlp\\), my little pony"],
    },
    "buwaro_elexion": {
        "character": ["buwaro_elexion"],
        "trigger": ["buwaro elexion, slightly damned"],
    },
    "krypto": {"character": ["krypto"], "trigger": ["krypto, krypto the superdog"]},
    "pepper_clark": {
        "character": ["pepper_clark"],
        "trigger": ["pepper clark, hasbro"],
    },
    "engineer_(team_fortress_2)": {
        "character": ["engineer_(team_fortress_2)"],
        "trigger": ["engineer \\(team fortress 2\\), valve"],
    },
    "yuguni_(yuguni)": {
        "character": ["yuguni_(yuguni)"],
        "trigger": ["yuguni \\(yuguni\\), converse"],
    },
    "boron_brioche": {
        "character": ["boron_brioche"],
        "trigger": ["boron brioche, fuga: melodies of steel"],
    },
    "rhea_snaketail": {
        "character": ["rhea_snaketail"],
        "trigger": ["rhea snaketail, slightly damned"],
    },
    "marc_(theblueberrycarrots)": {
        "character": ["marc_(theblueberrycarrots)"],
        "trigger": ["marc \\(theblueberrycarrots\\), inktober"],
    },
    "hiroshi_odokawa_(odd_taxi)": {
        "character": ["hiroshi_odokawa_(odd_taxi)"],
        "trigger": ["hiroshi odokawa \\(odd taxi\\), odd taxi"],
    },
    "tirek_(mlp)": {
        "character": ["tirek_(mlp)"],
        "trigger": ["tirek \\(mlp\\), my little pony"],
    },
    "junior_horse": {"character": ["junior_horse"], "trigger": ["junior horse, swat"]},
    "peter_pete_sr.": {
        "character": ["peter_pete_sr."],
        "trigger": ["peter pete sr., disney"],
    },
    "y'shtola_rhul": {
        "character": ["y'shtola_rhul"],
        "trigger": ["y'shtola rhul, square enix"],
    },
    "pato_(bastriw)": {
        "character": ["pato_(bastriw)"],
        "trigger": ["pato \\(bastriw\\), patreon"],
    },
    "sekiguchi_(odd_taxi)": {
        "character": ["sekiguchi_(odd_taxi)"],
        "trigger": ["sekiguchi \\(odd taxi\\), odd taxi"],
    },
    "princess_jasmine_(disney)": {
        "character": ["princess_jasmine_(disney)"],
        "trigger": ["princess jasmine \\(disney\\), disney"],
    },
    "stith": {"character": ["stith"], "trigger": ["stith, titan a.e."]},
    "lucy_(bcb)": {
        "character": ["lucy_(bcb)"],
        "trigger": ["lucy \\(bcb\\), bittersweet candy bowl"],
    },
    "farin": {"character": ["farin"], "trigger": ["farin, mythology"]},
    "trick_(tricktrashing)": {
        "character": ["trick_(tricktrashing)"],
        "trigger": ["trick \\(tricktrashing\\), nintendo"],
    },
    "gabu": {"character": ["gabu"], "trigger": ["gabu, one stormy night"]},
    "shere_khan": {"character": ["shere_khan"], "trigger": ["shere khan, disney"]},
    "catwoman": {"character": ["catwoman"], "trigger": ["catwoman, dc comics"]},
    "turanga_leela": {
        "character": ["turanga_leela"],
        "trigger": ["turanga leela, comedy central"],
    },
    "chelsea_chamberlain": {
        "character": ["chelsea_chamberlain"],
        "trigger": ["chelsea chamberlain, mythology"],
    },
    "francine_(animal_crossing)": {
        "character": ["francine_(animal_crossing)"],
        "trigger": ["francine \\(animal crossing\\), animal crossing"],
    },
    "coco_(animal_crossing)": {
        "character": ["coco_(animal_crossing)"],
        "trigger": ["coco \\(animal crossing\\), animal crossing"],
    },
    "lena_(ducktales)": {
        "character": ["lena_(ducktales)"],
        "trigger": ["lena \\(ducktales\\), disney"],
    },
    "medic_(team_fortress_2)": {
        "character": ["medic_(team_fortress_2)"],
        "trigger": ["medic \\(team fortress 2\\), valve"],
    },
    "fel_(my_life_with_fel)": {
        "character": ["fel_(my_life_with_fel)"],
        "trigger": ["fel \\(my life with fel\\), my life with fel"],
    },
    "kara_resch": {"character": ["kara_resch"], "trigger": ["kara resch, mythology"]},
    "pwink": {"character": ["pwink"], "trigger": ["pwink, mythology"]},
    "zatch_(notkastar)": {
        "character": ["zatch_(notkastar)"],
        "trigger": ["zatch \\(notkastar\\), nintendo"],
    },
    "collot_(beastars)": {
        "character": ["collot_(beastars)"],
        "trigger": ["collot \\(beastars\\), beastars"],
    },
    "freddy_(dislyte)": {
        "character": ["freddy_(dislyte)"],
        "trigger": ["freddy \\(dislyte\\), dislyte"],
    },
    "nepeta_leijon": {
        "character": ["nepeta_leijon"],
        "trigger": ["nepeta leijon, homestuck"],
    },
    "riptide_(riptideshark)": {
        "character": ["riptide_(riptideshark)"],
        "trigger": ["riptide \\(riptideshark\\), mythology"],
    },
    "penny_fenmore": {
        "character": ["penny_fenmore"],
        "trigger": ["penny fenmore, patreon"],
    },
    "mal0": {"character": ["mal0"], "trigger": ["mal0, scp foundation"]},
    "necrodrone_(character)": {
        "character": ["necrodrone_(character)"],
        "trigger": ["necrodrone \\(character\\), mythology"],
    },
    "clarice_(disney)": {
        "character": ["clarice_(disney)"],
        "trigger": ["clarice \\(disney\\), chip 'n dale"],
    },
    "carmen_(animal_crossing)": {
        "character": ["carmen_(animal_crossing)"],
        "trigger": ["carmen \\(animal crossing\\), animal crossing"],
    },
    "nicole_(nicnak044)": {
        "character": ["nicole_(nicnak044)"],
        "trigger": ["nicole \\(nicnak044\\), kodalynx"],
    },
    "blake_belladonna": {
        "character": ["blake_belladonna"],
        "trigger": ["blake belladonna, rwby"],
    },
    "ochaco_uraraka": {
        "character": ["ochaco_uraraka"],
        "trigger": ["ochaco uraraka, my hero academia"],
    },
    "taylor_renee_wolford_(darkflamewolf)": {
        "character": ["taylor_renee_wolford_(darkflamewolf)"],
        "trigger": ["taylor renee wolford \\(darkflamewolf\\), legend of ahya"],
    },
    "somnambula_(mlp)": {
        "character": ["somnambula_(mlp)"],
        "trigger": ["somnambula \\(mlp\\), my little pony"],
    },
    "mane-iac_(mlp)": {
        "character": ["mane-iac_(mlp)"],
        "trigger": ["mane-iac \\(mlp\\), my little pony"],
    },
    "renimpmon_x": {"character": ["renimpmon_x"], "trigger": ["renimpmon x, digimon"]},
    "koops": {"character": ["koops"], "trigger": ["koops, mario bros"]},
    "resine": {"character": ["resine"], "trigger": ["resine, mythology"]},
    "madam_reni_(twokinds)": {
        "character": ["madam_reni_(twokinds)"],
        "trigger": ["madam reni \\(twokinds\\), twokinds"],
    },
    "typhon_(tas)": {
        "character": ["typhon_(tas)"],
        "trigger": ["typhon \\(tas\\), lifewonders"],
    },
    "bernard_(ok_k.o.!_lbh)": {
        "character": ["bernard_(ok_k.o.!_lbh)"],
        "trigger": ["bernard \\(ok k.o.! lbh\\), cartoon network"],
    },
    "spirit_(cimarron)": {
        "character": ["spirit_(cimarron)"],
        "trigger": ["spirit \\(cimarron\\), dreamworks"],
    },
    "ami_dixie": {"character": ["ami_dixie"], "trigger": ["ami dixie, mythology"]},
    "inkling_boy": {"character": ["inkling_boy"], "trigger": ["inkling boy, splatoon"]},
    "rose_(funkybun)": {
        "character": ["rose_(funkybun)"],
        "trigger": ["rose \\(funkybun\\), halloween"],
    },
    "arcanis_(hahaluckyme)": {
        "character": ["arcanis_(hahaluckyme)"],
        "trigger": ["arcanis \\(hahaluckyme\\), nintendo"],
    },
    "luz_noceda": {"character": ["luz_noceda"], "trigger": ["luz noceda, disney"]},
    "commander_(commanderthings)": {
        "character": ["commander_(commanderthings)"],
        "trigger": ["commander \\(commanderthings\\), mythology"],
    },
    "dragaux": {"character": ["dragaux"], "trigger": ["dragaux, ring fit adventure"]},
    "malachi_(wooled)": {
        "character": ["malachi_(wooled)"],
        "trigger": ["malachi \\(wooled\\), pokemon mystery dungeon"],
    },
    "aria_blaze_(eg)": {
        "character": ["aria_blaze_(eg)"],
        "trigger": ["aria blaze \\(eg\\), my little pony"],
    },
    "patrick_star": {
        "character": ["patrick_star"],
        "trigger": ["patrick star, spongebob squarepants"],
    },
    "tangy_(animal_crossing)": {
        "character": ["tangy_(animal_crossing)"],
        "trigger": ["tangy \\(animal crossing\\), animal crossing"],
    },
    "vesairus": {"character": ["vesairus"], "trigger": ["vesairus, my little pony"]},
    "silver_(ezukapizumu)": {
        "character": ["silver_(ezukapizumu)"],
        "trigger": ["silver \\(ezukapizumu\\), mythology"],
    },
    "battle_principal_yuumi": {
        "character": ["battle_principal_yuumi"],
        "trigger": ["battle principal yuumi, riot games"],
    },
    "super_sonic": {
        "character": ["super_sonic"],
        "trigger": ["super sonic, sonic the hedgehog \\(series\\)"],
    },
    "sylvester": {
        "character": ["sylvester"],
        "trigger": ["sylvester, warner brothers"],
    },
    "anubislivess": {
        "character": ["anubislivess"],
        "trigger": ["anubislivess, mythology"],
    },
    "kutto": {"character": ["kutto"], "trigger": ["kutto, mythology"]},
    "malfaren": {"character": ["malfaren"], "trigger": ["malfaren, mythology"]},
    "bubba_(spyro)": {
        "character": ["bubba_(spyro)"],
        "trigger": ["bubba \\(spyro\\), mythology"],
    },
    "shirin_(bjekkergauken)": {
        "character": ["shirin_(bjekkergauken)"],
        "trigger": ["shirin \\(bjekkergauken\\)"],
    },
    "rain_silves": {
        "character": ["rain_silves"],
        "trigger": ["rain silves, clubstripes"],
    },
    "tod_(tfath)": {
        "character": ["tod_(tfath)"],
        "trigger": ["tod \\(tfath\\), disney"],
    },
    "mora": {"character": ["mora"], "trigger": ["mora, warfare machine"]},
    "tamati": {"character": ["tamati"], "trigger": ["tamati, mythology"]},
    "sophie_(argento)": {
        "character": ["sophie_(argento)"],
        "trigger": ["sophie \\(argento\\), sonic the hedgehog \\(series\\)"],
    },
    "docu_(doppel)": {
        "character": ["docu_(doppel)"],
        "trigger": ["docu \\(doppel\\), monster energy"],
    },
    "carla_(tcitw)": {
        "character": ["carla_(tcitw)"],
        "trigger": ["carla \\(tcitw\\), the cabin in the woods \\(arania\\)"],
    },
    "snake_(petruz)": {
        "character": ["snake_(petruz)"],
        "trigger": ["snake \\(petruz\\), petruz \\(copyright\\)"],
    },
    "pom-pom_(honkai:_star_rail)": {
        "character": ["pom-pom_(honkai:_star_rail)"],
        "trigger": ["pom-pom \\(honkai: star rail\\), honkai: star rail"],
    },
    "marbles_swiftfoot": {
        "character": ["marbles_swiftfoot"],
        "trigger": ["marbles swiftfoot, mythology"],
    },
    "dom_(naughtymorg)": {
        "character": ["dom_(naughtymorg)"],
        "trigger": ["dom \\(naughtymorg\\), mythology"],
    },
    "shirane_kan": {"character": ["shirane_kan"], "trigger": ["shirane kan, utau"]},
    "mei_(one_stormy_night)": {
        "character": ["mei_(one_stormy_night)"],
        "trigger": ["mei \\(one stormy night\\), one stormy night"],
    },
    "relle": {"character": ["relle"], "trigger": ["relle, mythology"]},
    "laylee": {"character": ["laylee"], "trigger": ["laylee, playtonic games"]},
    "selene_(pokemon)": {
        "character": ["selene_(pokemon)"],
        "trigger": ["selene \\(pokemon\\), pokemon"],
    },
    "alice_goldenfeather_(estories)": {
        "character": ["alice_goldenfeather_(estories)"],
        "trigger": ["alice goldenfeather \\(estories\\), my little pony"],
    },
    "squid_dog_(changed)": {
        "character": ["squid_dog_(changed)"],
        "trigger": ["squid dog \\(changed\\), changed \\(video game\\)"],
    },
    "pooh_bear": {"character": ["pooh_bear"], "trigger": ["pooh bear, disney"]},
    "xelthia": {"character": ["xelthia"], "trigger": ["xelthia, maya \\(software\\)"]},
    "rena_(yourdigimongirl)": {
        "character": ["rena_(yourdigimongirl)"],
        "trigger": ["rena \\(yourdigimongirl\\), digimon"],
    },
    "zhen_(kung_fu_panda)": {
        "character": ["zhen_(kung_fu_panda)"],
        "trigger": ["zhen \\(kung fu panda\\), kung fu panda"],
    },
    "maren_taverndatter": {
        "character": ["maren_taverndatter"],
        "trigger": ["maren taverndatter, twokinds"],
    },
    "aldea_(character)": {
        "character": ["aldea_(character)"],
        "trigger": ["aldea \\(character\\), christmas"],
    },
    "ty_conrad": {"character": ["ty_conrad"], "trigger": ["ty conrad, texnatsu"]},
    "blue_(blue's_clues)": {
        "character": ["blue_(blue's_clues)"],
        "trigger": ["blue \\(blue's clues\\), blue's clues"],
    },
    "sulley": {"character": ["sulley"], "trigger": ["sulley, disney"]},
    "mango_(3mangos)": {
        "character": ["mango_(3mangos)"],
        "trigger": ["mango \\(3mangos\\), my little pony"],
    },
    "excalibur_(zerofox)": {
        "character": ["excalibur_(zerofox)"],
        "trigger": ["excalibur \\(zerofox\\), mythology"],
    },
    "rocky_(paw_patrol)": {
        "character": ["rocky_(paw_patrol)"],
        "trigger": ["rocky \\(paw patrol\\), paw patrol"],
    },
    "samantha_brooks": {
        "character": ["samantha_brooks"],
        "trigger": ["samantha brooks, dragon's crown"],
    },
    "epona_(tloz)": {
        "character": ["epona_(tloz)"],
        "trigger": ["epona \\(tloz\\), the legend of zelda"],
    },
    "master_chief": {
        "character": ["master_chief"],
        "trigger": ["master chief, microsoft"],
    },
    "winona_(mlp)": {
        "character": ["winona_(mlp)"],
        "trigger": ["winona \\(mlp\\), my little pony"],
    },
    "timothy_vladislaus": {
        "character": ["timothy_vladislaus"],
        "trigger": ["timothy vladislaus, pokemon"],
    },
    "monokuma": {"character": ["monokuma"], "trigger": ["monokuma, danganronpa"]},
    "yamato_(one_piece)": {
        "character": ["yamato_(one_piece)"],
        "trigger": ["yamato \\(one piece\\), one piece"],
    },
    "slushi_(chikn_nuggit)": {
        "character": ["slushi_(chikn_nuggit)"],
        "trigger": ["slushi \\(chikn nuggit\\), chikn nuggit"],
    },
    "theodore_seville": {
        "character": ["theodore_seville"],
        "trigger": ["theodore seville, alvin and the chipmunks"],
    },
    "king_julien": {
        "character": ["king_julien"],
        "trigger": ["king julien, dreamworks"],
    },
    "pheagle": {"character": ["pheagle"], "trigger": ["pheagle, nfl"]},
    "dogmeat": {"character": ["dogmeat"], "trigger": ["dogmeat, fallout"]},
    "widowmaker_(overwatch)": {
        "character": ["widowmaker_(overwatch)"],
        "trigger": ["widowmaker \\(overwatch\\), overwatch"],
    },
    "amber_(snoot_game)": {
        "character": ["amber_(snoot_game)"],
        "trigger": ["amber \\(snoot game\\), cavemanon studios"],
    },
    "cham_cham": {
        "character": ["cham_cham"],
        "trigger": ["cham cham, samurai shodown"],
    },
    "maleficent": {"character": ["maleficent"], "trigger": ["maleficent, disney"]},
    "arcee": {"character": ["arcee"], "trigger": ["arcee, hasbro"]},
    "timbywuff": {"character": ["timbywuff"], "trigger": ["timbywuff, mythology"]},
    "betty_(weaver)": {
        "character": ["betty_(weaver)"],
        "trigger": ["betty \\(weaver\\), pack street"],
    },
    "furball_(character)": {
        "character": ["furball_(character)"],
        "trigger": ["furball \\(character\\), mythology"],
    },
    "set_(deity)": {
        "character": ["set_(deity)"],
        "trigger": ["set \\(deity\\), egyptian mythology"],
    },
    "diamondwing": {
        "character": ["diamondwing"],
        "trigger": ["diamondwing, mythology"],
    },
    "mighty_the_armadillo": {
        "character": ["mighty_the_armadillo"],
        "trigger": ["mighty the armadillo, sonic the hedgehog \\(series\\)"],
    },
    "olivia_(kadath)": {
        "character": ["olivia_(kadath)"],
        "trigger": ["olivia \\(kadath\\), patreon"],
    },
    "amethyst_(steven_universe)": {
        "character": ["amethyst_(steven_universe)"],
        "trigger": ["amethyst \\(steven universe\\), cartoon network"],
    },
    "snow_(tas)": {
        "character": ["snow_(tas)"],
        "trigger": ["snow \\(tas\\), lifewonders"],
    },
    "tek_(tekandprieda)": {
        "character": ["tek_(tekandprieda)"],
        "trigger": ["tek \\(tekandprieda\\), the rune tapper"],
    },
    "jasiri": {"character": ["jasiri"], "trigger": ["jasiri, the shadow of light"]},
    "agent_8_(splatoon)": {
        "character": ["agent_8_(splatoon)"],
        "trigger": ["agent 8 \\(splatoon\\), splatoon"],
    },
    "dr._k_(changed)": {
        "character": ["dr._k_(changed)"],
        "trigger": ["dr. k \\(changed\\), changed \\(video game\\)"],
    },
    "rue_(the-minuscule-task)": {
        "character": ["rue_(the-minuscule-task)"],
        "trigger": ["rue \\(the-minuscule-task\\), christmas"],
    },
    "nick_(the_xing1)": {
        "character": ["nick_(the_xing1)"],
        "trigger": ["nick \\(the xing1\\), easter"],
    },
    "brokenwing": {"character": ["brokenwing"], "trigger": ["brokenwing, mythology"]},
    "kibbles_(uberquest)": {
        "character": ["kibbles_(uberquest)"],
        "trigger": ["kibbles \\(uberquest\\), uberquest"],
    },
    "rin_kaenbyou": {
        "character": ["rin_kaenbyou"],
        "trigger": ["rin kaenbyou, touhou"],
    },
    "genesis_(kabscorner)": {
        "character": ["genesis_(kabscorner)"],
        "trigger": ["genesis \\(kabscorner\\), mythology"],
    },
    "king_snugglemagne_xxv": {
        "character": ["king_snugglemagne_xxv"],
        "trigger": ["king snugglemagne xxv, cartoon network"],
    },
    "sallie_may_(helluva_boss)": {
        "character": ["sallie_may_(helluva_boss)"],
        "trigger": ["sallie may \\(helluva boss\\), helluva boss"],
    },
    "kimahri": {"character": ["kimahri"], "trigger": ["kimahri, final fantasy x"]},
    "ayn_(fluff-kevlar)": {
        "character": ["ayn_(fluff-kevlar)"],
        "trigger": ["ayn \\(fluff-kevlar\\), patreon"],
    },
    "starfire_(teen_titans)": {
        "character": ["starfire_(teen_titans)"],
        "trigger": ["starfire \\(teen titans\\), dc comics"],
    },
    "shads": {"character": ["shads"], "trigger": ["shads, the shadow of light"]},
    "aolun_(character)": {
        "character": ["aolun_(character)"],
        "trigger": ["aolun \\(character\\), star fox"],
    },
    "leviathan_(skullgirls)": {
        "character": ["leviathan_(skullgirls)"],
        "trigger": ["leviathan \\(skullgirls\\), skullgirls"],
    },
    "kitsune_(ero)": {
        "character": ["kitsune_(ero)"],
        "trigger": ["kitsune \\(ero\\), nintendo"],
    },
    "aventis": {"character": ["aventis"], "trigger": ["aventis, mythology"]},
    "lucia_(satina)": {
        "character": ["lucia_(satina)"],
        "trigger": ["lucia \\(satina\\), satina wants a glass of water"],
    },
    "acta_(spacewaifu)": {
        "character": ["acta_(spacewaifu)"],
        "trigger": ["acta \\(spacewaifu\\), nintendo"],
    },
    "doggie_kruger": {
        "character": ["doggie_kruger"],
        "trigger": ["doggie kruger, power rangers"],
    },
    "elio_(pokemon)": {
        "character": ["elio_(pokemon)"],
        "trigger": ["elio \\(pokemon\\), pokemon"],
    },
    "sprig_plantar": {
        "character": ["sprig_plantar"],
        "trigger": ["sprig plantar, disney"],
    },
    "milo_(juantriforce)": {
        "character": ["milo_(juantriforce)"],
        "trigger": ["milo \\(juantriforce\\), mythology"],
    },
    "lemmy_koopa": {
        "character": ["lemmy_koopa"],
        "trigger": ["lemmy koopa, mario bros"],
    },
    "ezria": {"character": ["ezria"], "trigger": ["ezria, maya \\(software\\)"]},
    "discordnight": {
        "character": ["discordnight"],
        "trigger": ["discordnight, fallout"],
    },
    "anna_(kelnich)": {
        "character": ["anna_(kelnich)"],
        "trigger": ["anna \\(kelnich\\), my little pony"],
    },
    "angie_(study_partners)": {
        "character": ["angie_(study_partners)"],
        "trigger": ["angie \\(study partners\\), study partners"],
    },
    "denisse": {
        "character": ["denisse"],
        "trigger": ["denisse, blender \\(software\\)"],
    },
    "lillia_(lol)": {
        "character": ["lillia_(lol)"],
        "trigger": ["lillia \\(lol\\), riot games"],
    },
    "glamrock_bonnie_(fnaf)": {
        "character": ["glamrock_bonnie_(fnaf)"],
        "trigger": ["glamrock bonnie \\(fnaf\\), scottgames"],
    },
    "sherly_karu": {
        "character": ["sherly_karu"],
        "trigger": ["sherly karu, mythology"],
    },
    "charle_(fairy_tail)": {
        "character": ["charle_(fairy_tail)"],
        "trigger": ["charle \\(fairy tail\\), fairy tail"],
    },
    "photo_finish_(mlp)": {
        "character": ["photo_finish_(mlp)"],
        "trigger": ["photo finish \\(mlp\\), my little pony"],
    },
    "substitute_doll": {
        "character": ["substitute_doll"],
        "trigger": ["substitute doll, pokemon"],
    },
    "charlie_(weaver)": {
        "character": ["charlie_(weaver)"],
        "trigger": ["charlie \\(weaver\\), pack street"],
    },
    "impostor_(among_us)": {
        "character": ["impostor_(among_us)"],
        "trigger": ["impostor \\(among us\\), among us"],
    },
    "cheezborger_(chikn_nuggit)": {
        "character": ["cheezborger_(chikn_nuggit)"],
        "trigger": ["cheezborger \\(chikn nuggit\\), chikn nuggit"],
    },
    "smolder": {"character": ["smolder"], "trigger": ["smolder, mythology"]},
    "lee_(ajdurai)": {
        "character": ["lee_(ajdurai)"],
        "trigger": ["lee \\(ajdurai\\), mythology"],
    },
    "nightdancer_(character)": {
        "character": ["nightdancer_(character)"],
        "trigger": ["nightdancer \\(character\\), disney"],
    },
    "elias_ainsworth": {
        "character": ["elias_ainsworth"],
        "trigger": ["elias ainsworth, the ancient magus bride"],
    },
    "ismar": {"character": ["ismar"], "trigger": ["ismar, mythology"]},
    "faputa": {"character": ["faputa"], "trigger": ["faputa, made in abyss"]},
    "frieren": {
        "character": ["frieren"],
        "trigger": ["frieren, frieren beyond journey's end"],
    },
    "bagheera_(jungle_book)": {
        "character": ["bagheera_(jungle_book)"],
        "trigger": ["bagheera \\(jungle book\\), the jungle book"],
    },
    "rufus_black": {
        "character": ["rufus_black"],
        "trigger": ["rufus black, mythology"],
    },
    "furx_(character)": {
        "character": ["furx_(character)"],
        "trigger": ["furx \\(character\\), mythology"],
    },
    "slenderman": {"character": ["slenderman"], "trigger": ["slenderman, creepypasta"]},
    "doge": {"character": ["doge"], "trigger": ["doge, dogelore"]},
    "hugh_muskroura": {
        "character": ["hugh_muskroura"],
        "trigger": ["hugh muskroura, disney"],
    },
    "ahya_(legend_of_ahya)": {
        "character": ["ahya_(legend_of_ahya)"],
        "trigger": ["ahya \\(legend of ahya\\), legend of ahya"],
    },
    "sergen_(silver_soul)": {
        "character": ["sergen_(silver_soul)"],
        "trigger": ["sergen \\(silver soul\\), pokemon"],
    },
    "moji_(paladins)": {
        "character": ["moji_(paladins)"],
        "trigger": ["moji \\(paladins\\), paladins \\(game\\)"],
    },
    "inigo_(kusosensei)": {
        "character": ["inigo_(kusosensei)"],
        "trigger": ["inigo \\(kusosensei\\), halloween"],
    },
    "ciena_celle": {
        "character": ["ciena_celle"],
        "trigger": ["ciena celle, mythology"],
    },
    "axo_(fortnite)": {
        "character": ["axo_(fortnite)"],
        "trigger": ["axo \\(fortnite\\), fortnite"],
    },
    "petri_(animal_crossing)": {
        "character": ["petri_(animal_crossing)"],
        "trigger": ["petri \\(animal crossing\\), animal crossing"],
    },
    "proxy_(pizzacat)": {
        "character": ["proxy_(pizzacat)"],
        "trigger": ["proxy \\(pizzacat\\), samurai pizza cats"],
    },
    "lauren_faust_(character)": {
        "character": ["lauren_faust_(character)"],
        "trigger": ["lauren faust \\(character\\), my little pony"],
    },
    "mr._peabody": {
        "character": ["mr._peabody"],
        "trigger": ["mr. peabody, mr. peabody and sherman"],
    },
    "lapis_lazuli_(steven_universe)": {
        "character": ["lapis_lazuli_(steven_universe)"],
        "trigger": ["lapis lazuli \\(steven universe\\), cartoon network"],
    },
    "rubber_lass": {"character": ["rubber_lass"], "trigger": ["rubber lass, nintendo"]},
    "julie-su": {
        "character": ["julie-su"],
        "trigger": ["julie-su, sonic the hedgehog \\(series\\)"],
    },
    "peter_the_cat": {
        "character": ["peter_the_cat"],
        "trigger": ["peter the cat, nintendo"],
    },
    "jakethegoat_(character)": {
        "character": ["jakethegoat_(character)"],
        "trigger": ["jakethegoat \\(character\\), mythology"],
    },
    "fran_(litterbox_comics)": {
        "character": ["fran_(litterbox_comics)"],
        "trigger": ["fran \\(litterbox comics\\), litterbox comics"],
    },
    "maci_(ceehaz)": {
        "character": ["maci_(ceehaz)"],
        "trigger": ["maci \\(ceehaz\\), dog knight rpg"],
    },
    "cupid": {"character": ["cupid"], "trigger": ["cupid, valentine's day"]},
    "charmy_bee": {
        "character": ["charmy_bee"],
        "trigger": ["charmy bee, sonic the hedgehog \\(series\\)"],
    },
    "flame_princess": {
        "character": ["flame_princess"],
        "trigger": ["flame princess, cartoon network"],
    },
    "jaki-kun_(character)": {
        "character": ["jaki-kun_(character)"],
        "trigger": ["jaki-kun \\(character\\), pokemon"],
    },
    "tokugawa_ieyasu": {
        "character": ["tokugawa_ieyasu"],
        "trigger": ["tokugawa ieyasu, sengoku puzzle"],
    },
    "blitz_(gyro)": {
        "character": ["blitz_(gyro)"],
        "trigger": ["blitz \\(gyro\\), pokemon"],
    },
    "ivy_valentine": {
        "character": ["ivy_valentine"],
        "trigger": ["ivy valentine, soul calibur"],
    },
    "jill_valentine": {
        "character": ["jill_valentine"],
        "trigger": ["jill valentine, resident evil"],
    },
    "balloon_boy_(fnaf)": {
        "character": ["balloon_boy_(fnaf)"],
        "trigger": ["balloon boy \\(fnaf\\), five nights at freddy's 2"],
    },
    "chance_the_rabbit": {
        "character": ["chance_the_rabbit"],
        "trigger": ["chance the rabbit, divine acid"],
    },
    "herja_(bjekkergauken)": {
        "character": ["herja_(bjekkergauken)"],
        "trigger": ["herja \\(bjekkergauken\\)"],
    },
    "nani_pelekai": {
        "character": ["nani_pelekai"],
        "trigger": ["nani pelekai, disney"],
    },
    "dib_membrane": {
        "character": ["dib_membrane"],
        "trigger": ["dib membrane, invader zim"],
    },
    "queen_sectonia": {
        "character": ["queen_sectonia"],
        "trigger": ["queen sectonia, kirby \\(series\\)"],
    },
    "matt_riskely": {
        "character": ["matt_riskely"],
        "trigger": ["matt riskely, christmas"],
    },
    "solid_snake": {
        "character": ["solid_snake"],
        "trigger": ["solid snake, metal gear"],
    },
    "wonder_woman": {
        "character": ["wonder_woman"],
        "trigger": ["wonder woman, dc comics"],
    },
    "chocolat_gelato": {
        "character": ["chocolat_gelato"],
        "trigger": ["chocolat gelato, solatorobo"],
    },
    "norbert_beaver": {
        "character": ["norbert_beaver"],
        "trigger": ["norbert beaver, the angry beavers"],
    },
    "dulce_(mr.pink)": {
        "character": ["dulce_(mr.pink)"],
        "trigger": ["dulce \\(mr.pink\\), valentine's day"],
    },
    "asbie": {"character": ["asbie"], "trigger": ["asbie, pokemon"]},
    "inuki_zu": {"character": ["inuki_zu"], "trigger": ["inuki zu, nintendo"]},
    "kili_(kilinah)": {
        "character": ["kili_(kilinah)"],
        "trigger": ["kili \\(kilinah\\), mythology"],
    },
    "maid_marian_(qnaoz)": {
        "character": ["maid_marian_(qnaoz)"],
        "trigger": ["maid marian \\(qnaoz\\), disney"],
    },
    "destiney_crawford_(thatworgen)": {
        "character": ["destiney_crawford_(thatworgen)"],
        "trigger": ["destiney crawford \\(thatworgen\\), warcraft"],
    },
    "fernier": {"character": ["fernier"], "trigger": ["fernier, warcraft"]},
    "pang_(sdorica)": {
        "character": ["pang_(sdorica)"],
        "trigger": ["pang \\(sdorica\\), sdorica"],
    },
    "yukiminus_rex_(evov1)": {
        "character": ["yukiminus_rex_(evov1)"],
        "trigger": ["yukiminus rex \\(evov1\\), universal studios"],
    },
    "dog_operator": {
        "character": ["dog_operator"],
        "trigger": ["dog operator, lifewonders"],
    },
    "kathy_(danellz)": {
        "character": ["kathy_(danellz)"],
        "trigger": ["kathy \\(danellz\\), capcom"],
    },
    "sovy": {"character": ["sovy"], "trigger": ["sovy, mythology"]},
    "sash_(backsash)": {
        "character": ["sash_(backsash)"],
        "trigger": ["sash \\(backsash\\), nintendo"],
    },
    "fur_(theterm)": {
        "character": ["fur_(theterm)"],
        "trigger": ["fur \\(theterm\\), jojo's bizarre adventure"],
    },
    "lt._john_llama": {
        "character": ["lt._john_llama"],
        "trigger": ["lt. john llama, fortnite"],
    },
    "ruby_(jewelpet)": {
        "character": ["ruby_(jewelpet)"],
        "trigger": ["ruby \\(jewelpet\\), jewelpet"],
    },
    "alakay_alex": {
        "character": ["alakay_alex"],
        "trigger": ["alakay alex, madagascar \\(series\\)"],
    },
    "jcfox": {"character": ["jcfox"], "trigger": ["jcfox, sony corporation"]},
    "mylar_(discreet_user)": {
        "character": ["mylar_(discreet_user)"],
        "trigger": ["mylar \\(discreet user\\), mythology"],
    },
    "charlie_barkin": {
        "character": ["charlie_barkin"],
        "trigger": ["charlie barkin, don bluth"],
    },
    "rikki": {"character": ["rikki"], "trigger": ["rikki, patreon"]},
    "azure_(bluedude)": {
        "character": ["azure_(bluedude)"],
        "trigger": ["azure \\(bluedude\\), mythology"],
    },
    "kisha": {"character": ["kisha"], "trigger": ["kisha, christmas"]},
    "lucifer_(helltaker)": {
        "character": ["lucifer_(helltaker)"],
        "trigger": ["lucifer \\(helltaker\\), helltaker"],
    },
    "ranshiin": {"character": ["ranshiin"], "trigger": ["ranshiin, legendz"]},
    "ariel_(disney)": {
        "character": ["ariel_(disney)"],
        "trigger": ["ariel \\(disney\\), disney"],
    },
    "georgette_(disney)": {
        "character": ["georgette_(disney)"],
        "trigger": ["georgette \\(disney\\), disney"],
    },
    "ru_(rudragon)": {
        "character": ["ru_(rudragon)"],
        "trigger": ["ru \\(rudragon\\), mythology"],
    },
    "loimu_(character)": {
        "character": ["loimu_(character)"],
        "trigger": ["loimu \\(character\\), mythology"],
    },
    "katt_(breath_of_fire)": {
        "character": ["katt_(breath_of_fire)"],
        "trigger": ["katt \\(breath of fire\\), breath of fire"],
    },
    "mia_(.hack)": {
        "character": ["mia_(.hack)"],
        "trigger": ["mia \\(.hack\\), .hack"],
    },
    "reptile_(mortal_kombat)": {
        "character": ["reptile_(mortal_kombat)"],
        "trigger": ["reptile \\(mortal kombat\\), mortal kombat"],
    },
    "mei_(overwatch)": {
        "character": ["mei_(overwatch)"],
        "trigger": ["mei \\(overwatch\\), overwatch"],
    },
    "warfare_carmelita": {
        "character": ["warfare_carmelita"],
        "trigger": ["warfare carmelita, sony interactive entertainment"],
    },
    "checker": {"character": ["checker"], "trigger": ["checker, pokemon"]},
    "hellhound_(mge)": {
        "character": ["hellhound_(mge)"],
        "trigger": ["hellhound \\(mge\\), monster girl encyclopedia"],
    },
    "rysoka": {"character": ["rysoka"], "trigger": ["rysoka, mythology"]},
    "wilford_wolf": {
        "character": ["wilford_wolf"],
        "trigger": ["wilford wolf, warner brothers"],
    },
    "cox": {"character": ["cox"], "trigger": ["cox, clubstripes"]},
    "miyamoto_usagi": {
        "character": ["miyamoto_usagi"],
        "trigger": ["miyamoto usagi, usagi yojimbo"],
    },
    "vixey_(tfath)": {
        "character": ["vixey_(tfath)"],
        "trigger": ["vixey \\(tfath\\), disney"],
    },
    "miss_kitty_mouse": {
        "character": ["miss_kitty_mouse"],
        "trigger": ["miss kitty mouse, disney"],
    },
    "roni_collins": {
        "character": ["roni_collins"],
        "trigger": ["roni collins, mythology"],
    },
    "marx_(kirby)": {
        "character": ["marx_(kirby)"],
        "trigger": ["marx \\(kirby\\), kirby \\(series\\)"],
    },
    "fizzle_(mlp)": {
        "character": ["fizzle_(mlp)"],
        "trigger": ["fizzle \\(mlp\\), my little pony"],
    },
    "foxy-rena": {"character": ["foxy-rena"], "trigger": ["foxy-rena, mythology"]},
    "rasha": {"character": ["rasha"], "trigger": ["rasha, nintendo"]},
    "shaze": {"character": ["shaze"], "trigger": ["shaze, nintendo"]},
    "nolegs_(oc)": {
        "character": ["nolegs_(oc)"],
        "trigger": ["nolegs \\(oc\\), my little pony"],
    },
    "rj_oakes": {"character": ["rj_oakes"], "trigger": ["rj oakes, h.w.t. studios"]},
    "nia_(xenoblade)": {
        "character": ["nia_(xenoblade)"],
        "trigger": ["nia \\(xenoblade\\), xenoblade \\(series\\)"],
    },
    "syl_(enginetrap)": {
        "character": ["syl_(enginetrap)"],
        "trigger": ["syl \\(enginetrap\\), mythology"],
    },
    "melon_(beastars)": {
        "character": ["melon_(beastars)"],
        "trigger": ["melon \\(beastars\\), beastars"],
    },
    "mowgli": {"character": ["mowgli"], "trigger": ["mowgli, the jungle book"]},
    "morton_koopa_jr.": {
        "character": ["morton_koopa_jr."],
        "trigger": ["morton koopa jr., mario bros"],
    },
    "magica_de_spell": {
        "character": ["magica_de_spell"],
        "trigger": ["magica de spell, disney"],
    },
    "derrick_(hextra)": {
        "character": ["derrick_(hextra)"],
        "trigger": ["derrick \\(hextra\\), hextra"],
    },
    "hal_greaves": {"character": ["hal_greaves"], "trigger": ["hal greaves, group17"]},
    "silvia_windmane": {
        "character": ["silvia_windmane"],
        "trigger": ["silvia windmane, my little pony"],
    },
    "winston_(overwatch)": {
        "character": ["winston_(overwatch)"],
        "trigger": ["winston \\(overwatch\\), overwatch"],
    },
    "haloren": {"character": ["haloren"], "trigger": ["haloren, mythology"]},
    "cregon": {"character": ["cregon"], "trigger": ["cregon, ford"]},
    "ikasama": {
        "character": ["ikasama"],
        "trigger": ["ikasama, gamba no bouken \\(series\\)"],
    },
    "scrooge_mcduck": {
        "character": ["scrooge_mcduck"],
        "trigger": ["scrooge mcduck, disney"],
    },
    "blu_(rio)": {
        "character": ["blu_(rio)"],
        "trigger": ["blu \\(rio\\), blue sky studios"],
    },
    "richard_(james_howard)": {
        "character": ["richard_(james_howard)"],
        "trigger": ["richard \\(james howard\\), patreon"],
    },
    "neneruko_(doneru)": {
        "character": ["neneruko_(doneru)"],
        "trigger": ["neneruko \\(doneru\\), mythology"],
    },
    "unni_(bjekkergauken)": {
        "character": ["unni_(bjekkergauken)"],
        "trigger": ["unni \\(bjekkergauken\\)"],
    },
    "kodiak_(balto)": {
        "character": ["kodiak_(balto)"],
        "trigger": ["kodiak \\(balto\\), universal studios"],
    },
    "kate_(alpha_and_omega)": {
        "character": ["kate_(alpha_and_omega)"],
        "trigger": ["kate \\(alpha and omega\\), alpha and omega"],
    },
    "tycloud": {"character": ["tycloud"], "trigger": ["tycloud, nintendo"]},
    "jasiri_(the_lion_guard)": {
        "character": ["jasiri_(the_lion_guard)"],
        "trigger": ["jasiri \\(the lion guard\\), disney"],
    },
    "rennin": {"character": ["rennin"], "trigger": ["rennin, mythology"]},
    "duster_(dustafyer7)": {
        "character": ["duster_(dustafyer7)"],
        "trigger": ["duster \\(dustafyer7\\), mythology"],
    },
    "rachnera_arachnera_(monster_musume)": {
        "character": ["rachnera_arachnera_(monster_musume)"],
        "trigger": ["rachnera arachnera \\(monster musume\\), monster musume"],
    },
    "fuli": {"character": ["fuli"], "trigger": ["fuli, disney"]},
    "kali_belladonna": {
        "character": ["kali_belladonna"],
        "trigger": ["kali belladonna, rwby"],
    },
    "fernando_(bastriw)": {
        "character": ["fernando_(bastriw)"],
        "trigger": ["fernando \\(bastriw\\), patreon"],
    },
    "kimba": {"character": ["kimba"], "trigger": ["kimba, osamu tezuka"]},
    "don_karnage": {"character": ["don_karnage"], "trigger": ["don karnage, disney"]},
    "rabbit_(winnie_the_pooh)": {
        "character": ["rabbit_(winnie_the_pooh)"],
        "trigger": ["rabbit \\(winnie the pooh\\), disney"],
    },
    "tiny_tiger": {
        "character": ["tiny_tiger"],
        "trigger": ["tiny tiger, crash bandicoot \\(series\\)"],
    },
    "lilith_(zajice)": {
        "character": ["lilith_(zajice)"],
        "trigger": ["lilith \\(zajice\\), my little pony"],
    },
    "dust_(elysian_tail)": {
        "character": ["dust_(elysian_tail)"],
        "trigger": ["dust \\(elysian tail\\), dust: an elysian tail"],
    },
    "dipper_pines": {
        "character": ["dipper_pines"],
        "trigger": ["dipper pines, disney"],
    },
    "stripe_heeler": {
        "character": ["stripe_heeler"],
        "trigger": ["stripe heeler, bluey \\(series\\)"],
    },
    "tarnished_(elden_ring)": {
        "character": ["tarnished_(elden_ring)"],
        "trigger": ["tarnished \\(elden ring\\), fromsoftware"],
    },
    "onyx_(jmh)": {
        "character": ["onyx_(jmh)"],
        "trigger": ["onyx \\(jmh\\), christmas"],
    },
    "drayk_dagger": {
        "character": ["drayk_dagger"],
        "trigger": ["drayk dagger, mythology"],
    },
    "scotty_kat": {"character": ["scotty_kat"], "trigger": ["scotty kat, mythology"]},
    "adira_riftwall": {
        "character": ["adira_riftwall"],
        "trigger": ["adira riftwall, twokinds"],
    },
    "nightmare_foxy_(fnaf)": {
        "character": ["nightmare_foxy_(fnaf)"],
        "trigger": ["nightmare foxy \\(fnaf\\), scottgames"],
    },
    "elise_(greyhunter)": {
        "character": ["elise_(greyhunter)"],
        "trigger": ["elise \\(greyhunter\\), christmas"],
    },
    "secretary_washimi": {
        "character": ["secretary_washimi"],
        "trigger": ["secretary washimi, sanrio"],
    },
    "pink_(pink)": {
        "character": ["pink_(pink)"],
        "trigger": ["pink \\(pink\\), da silva"],
    },
    "crimvael_(interspecies_reviewers)": {
        "character": ["crimvael_(interspecies_reviewers)"],
        "trigger": ["crimvael \\(interspecies reviewers\\), interspecies reviewers"],
    },
    "helm_(connivingrat)": {
        "character": ["helm_(connivingrat)"],
        "trigger": ["helm \\(connivingrat\\), mythology"],
    },
    "jose_carioca": {
        "character": ["jose_carioca"],
        "trigger": ["jose carioca, disney"],
    },
    "kevin_snowpaw": {
        "character": ["kevin_snowpaw"],
        "trigger": ["kevin snowpaw, mythology"],
    },
    "plushtrap_(fnaf)": {
        "character": ["plushtrap_(fnaf)"],
        "trigger": ["plushtrap \\(fnaf\\), scottgames"],
    },
    "kai_the_collector": {
        "character": ["kai_the_collector"],
        "trigger": ["kai the collector, kung fu panda"],
    },
    "zephyr_breeze_(mlp)": {
        "character": ["zephyr_breeze_(mlp)"],
        "trigger": ["zephyr breeze \\(mlp\\), my little pony"],
    },
    "yuki_yoshida": {
        "character": ["yuki_yoshida"],
        "trigger": ["yuki yoshida, cartoon network"],
    },
    "arteia_kincaid_(arctic_android)": {
        "character": ["arteia_kincaid_(arctic_android)"],
        "trigger": ["arteia kincaid \\(arctic android\\), bunny and fox world"],
    },
    "sniper_(team_fortress_2)": {
        "character": ["sniper_(team_fortress_2)"],
        "trigger": ["sniper \\(team fortress 2\\), valve"],
    },
    "miss_l": {"character": ["miss_l"], "trigger": ["miss l, mythology"]},
    "tyrande_whisperwind": {
        "character": ["tyrande_whisperwind"],
        "trigger": ["tyrande whisperwind, warcraft"],
    },
    "kieran": {"character": ["kieran"], "trigger": ["kieran, mythology"]},
    "zooey_the_fox": {
        "character": ["zooey_the_fox"],
        "trigger": ["zooey the fox, sonic the hedgehog \\(series\\)"],
    },
    "sitri_(tas)": {
        "character": ["sitri_(tas)"],
        "trigger": ["sitri \\(tas\\), lifewonders"],
    },
    "kalita_(furryfight_chronicles)": {
        "character": ["kalita_(furryfight_chronicles)"],
        "trigger": ["kalita \\(furryfight chronicles\\), furryfight chronicles"],
    },
    "ratih_(study_partners)": {
        "character": ["ratih_(study_partners)"],
        "trigger": ["ratih \\(study partners\\), study partners"],
    },
    "procy": {"character": ["procy"], "trigger": ["procy, lifewonders"]},
    "dashwood_fox": {
        "character": ["dashwood_fox"],
        "trigger": ["dashwood fox, nintendo"],
    },
    "valkyr_(warframe)": {
        "character": ["valkyr_(warframe)"],
        "trigger": ["valkyr \\(warframe\\), warframe"],
    },
    "superia": {"character": ["superia"], "trigger": ["superia, sega"]},
    "percy_vison": {"character": ["percy_vison"], "trigger": ["percy vison, disney"]},
    "lumikin": {"character": ["lumikin"], "trigger": ["lumikin, mythology"]},
    "swiper_(dora_the_explorer)": {
        "character": ["swiper_(dora_the_explorer)"],
        "trigger": ["swiper \\(dora the explorer\\), dora the explorer"],
    },
    "rocksteady": {
        "character": ["rocksteady"],
        "trigger": ["rocksteady, teenage mutant ninja turtles"],
    },
    "sino_(furfragged)": {
        "character": ["sino_(furfragged)"],
        "trigger": ["sino \\(furfragged\\), mythology"],
    },
    "jay-r_(character)": {
        "character": ["jay-r_(character)"],
        "trigger": ["jay-r \\(character\\), nintendo"],
    },
    "slappy_squirrel": {
        "character": ["slappy_squirrel"],
        "trigger": ["slappy squirrel, warner brothers"],
    },
    "frieza": {"character": ["frieza"], "trigger": ["frieza, dragon ball"]},
    "nhala_levee": {
        "character": ["nhala_levee"],
        "trigger": ["nhala levee, mythology"],
    },
    "phantasma_(ghoul_school)": {
        "character": ["phantasma_(ghoul_school)"],
        "trigger": ["phantasma \\(ghoul school\\), ghoul school"],
    },
    "wii_fit_trainer": {
        "character": ["wii_fit_trainer"],
        "trigger": ["wii fit trainer, wii fit"],
    },
    "condom_crusader": {
        "character": ["condom_crusader"],
        "trigger": ["condom crusader, nintendo"],
    },
    "fekkri_talot": {
        "character": ["fekkri_talot"],
        "trigger": ["fekkri talot, wizards of the coast"],
    },
    "raphtalia": {
        "character": ["raphtalia"],
        "trigger": ["raphtalia, the rising of the shield hero"],
    },
    "della_duck": {"character": ["della_duck"], "trigger": ["della duck, disney"]},
    "meilin_lee_(turning_red)": {
        "character": ["meilin_lee_(turning_red)"],
        "trigger": ["meilin lee \\(turning red\\), disney"],
    },
    "bitter_(bristol)": {
        "character": ["bitter_(bristol)"],
        "trigger": ["bitter \\(bristol\\), wildstar"],
    },
    "little_john": {"character": ["little_john"], "trigger": ["little john, disney"]},
    "malefor": {"character": ["malefor"], "trigger": ["malefor, spyro the dragon"]},
    "goliath_(gargoyles)": {
        "character": ["goliath_(gargoyles)"],
        "trigger": ["goliath \\(gargoyles\\), disney"],
    },
    "krauti_mercedes": {
        "character": ["krauti_mercedes"],
        "trigger": ["krauti mercedes, mythology"],
    },
    "squigga": {"character": ["squigga"], "trigger": ["squigga, big bun burgers"]},
    "atticus_mura": {
        "character": ["atticus_mura"],
        "trigger": ["atticus mura, pokemon"],
    },
    "thunder_(fortnite)": {
        "character": ["thunder_(fortnite)"],
        "trigger": ["thunder \\(fortnite\\), fortnite"],
    },
    "nami_(one_piece)": {
        "character": ["nami_(one_piece)"],
        "trigger": ["nami \\(one piece\\), one piece"],
    },
    "rammy_aaron": {
        "character": ["rammy_aaron"],
        "trigger": ["rammy aaron, mythology"],
    },
    "duchess_(aristocats)": {
        "character": ["duchess_(aristocats)"],
        "trigger": ["duchess \\(aristocats\\), disney"],
    },
    "wario": {"character": ["wario"], "trigger": ["wario, mario bros"]},
    "sparx": {"character": ["sparx"], "trigger": ["sparx, spyro the dragon"]},
    "sylvia_marpole": {
        "character": ["sylvia_marpole"],
        "trigger": ["sylvia marpole, disney"],
    },
    "king_kong": {"character": ["king_kong"], "trigger": ["king kong, toho"]},
    "loki_(bitterplaguerat)": {
        "character": ["loki_(bitterplaguerat)"],
        "trigger": ["loki \\(bitterplaguerat\\), my little pony"],
    },
    "lololo": {"character": ["lololo"], "trigger": ["lololo, kirby \\(series\\)"]},
    "nightswing": {"character": ["nightswing"], "trigger": ["nightswing, mythology"]},
    "dielle_(wooled)": {
        "character": ["dielle_(wooled)"],
        "trigger": ["dielle \\(wooled\\), pokemon"],
    },
    "four_arms_(ben_10)": {
        "character": ["four_arms_(ben_10)"],
        "trigger": ["four arms \\(ben 10\\), cartoon network"],
    },
    "eleanor_miller": {
        "character": ["eleanor_miller"],
        "trigger": ["eleanor miller, alvin and the chipmunks"],
    },
    "duga_(shining)": {
        "character": ["duga_(shining)"],
        "trigger": ["duga \\(shining\\), sega"],
    },
    "tala_(suntattoowolf)": {
        "character": ["tala_(suntattoowolf)"],
        "trigger": ["tala \\(suntattoowolf\\), mythology"],
    },
    "withered_chica_(fnaf)": {
        "character": ["withered_chica_(fnaf)"],
        "trigger": ["withered chica \\(fnaf\\), five nights at freddy's 2"],
    },
    "steven_quartz_universe": {
        "character": ["steven_quartz_universe"],
        "trigger": ["steven quartz universe, steven universe"],
    },
    "mrs.mayhem": {
        "character": ["mrs.mayhem"],
        "trigger": ["mrs.mayhem, super planet dolan"],
    },
    "keiko_sakmat": {
        "character": ["keiko_sakmat"],
        "trigger": ["keiko sakmat, christmas"],
    },
    "artificer_(rain_world)": {
        "character": ["artificer_(rain_world)"],
        "trigger": ["artificer \\(rain world\\), videocult"],
    },
    "rika_(rika)": {
        "character": ["rika_(rika)"],
        "trigger": ["rika \\(rika\\), new year"],
    },
    "kyubey": {
        "character": ["kyubey"],
        "trigger": ["kyubey, puella magi madoka magica"],
    },
    "oselotti_(character)": {
        "character": ["oselotti_(character)"],
        "trigger": ["oselotti \\(character\\), mythology"],
    },
    "sam_(kuroodod)": {
        "character": ["sam_(kuroodod)"],
        "trigger": ["sam \\(kuroodod\\), pokemon"],
    },
    "puppet_bonnie_(fnafsl)": {
        "character": ["puppet_bonnie_(fnafsl)"],
        "trigger": ["puppet bonnie \\(fnafsl\\), scottgames"],
    },
    "ghislaine_dedoldia": {
        "character": ["ghislaine_dedoldia"],
        "trigger": ["ghislaine dedoldia, mushoku tensei"],
    },
    "hobbes": {"character": ["hobbes"], "trigger": ["hobbes, calvin and hobbes"]},
    "shiro_uzumaki": {
        "character": ["shiro_uzumaki"],
        "trigger": ["shiro uzumaki, nintendo"],
    },
    "wynter": {"character": ["wynter"], "trigger": ["wynter, mythology"]},
    "ben_(roanoak)": {
        "character": ["ben_(roanoak)"],
        "trigger": ["ben \\(roanoak\\), patreon"],
    },
    "polt_(monster_musume)": {
        "character": ["polt_(monster_musume)"],
        "trigger": ["polt \\(monster musume\\), monster musume"],
    },
    "cervina": {"character": ["cervina"], "trigger": ["cervina, christmas"]},
    "furlong_(live_a_hero)": {
        "character": ["furlong_(live_a_hero)"],
        "trigger": ["furlong \\(live a hero\\), lifewonders"],
    },
    "hillevi": {"character": ["hillevi"], "trigger": ["hillevi"]},
    "dragoneer_(character)": {
        "character": ["dragoneer_(character)"],
        "trigger": ["dragoneer \\(character\\), digimon"],
    },
    "angel_kryis": {
        "character": ["angel_kryis"],
        "trigger": ["angel kryis, mythology"],
    },
    "drossel_von_flugel_(fireball)": {
        "character": ["drossel_von_flugel_(fireball)"],
        "trigger": ["drossel von flugel \\(fireball\\), disney"],
    },
    "nyaaa_foxx": {"character": ["nyaaa_foxx"], "trigger": ["nyaaa foxx, nintendo"]},
    "green_shadow": {
        "character": ["green_shadow"],
        "trigger": ["green shadow, plants vs. zombies heroes"],
    },
    "manitka_(character)": {
        "character": ["manitka_(character)"],
        "trigger": ["manitka \\(character\\), mythology"],
    },
    "fidget_the_fox": {
        "character": ["fidget_the_fox"],
        "trigger": ["fidget the fox, telegram"],
    },
    "modeus_(helltaker)": {
        "character": ["modeus_(helltaker)"],
        "trigger": ["modeus \\(helltaker\\), helltaker"],
    },
    "steve_(minecraft)": {
        "character": ["steve_(minecraft)"],
        "trigger": ["steve \\(minecraft\\), microsoft"],
    },
    "ayden_(brogulls)": {
        "character": ["ayden_(brogulls)"],
        "trigger": ["ayden \\(brogulls\\), brogulls"],
    },
    "bunnicula": {
        "character": ["bunnicula"],
        "trigger": ["bunnicula, bunnicula \\(series\\)"],
    },
    "dan_darkheart": {
        "character": ["dan_darkheart"],
        "trigger": ["dan darkheart, nintendo"],
    },
    "dj_bop": {"character": ["dj_bop"], "trigger": ["dj bop, fortnite"]},
    "pork_(cerealharem)": {
        "character": ["pork_(cerealharem)"],
        "trigger": ["pork \\(cerealharem\\), sailor moon \\(series\\)"],
    },
    "vegeta": {"character": ["vegeta"], "trigger": ["vegeta, dragon ball"]},
    "marge_simpson": {
        "character": ["marge_simpson"],
        "trigger": ["marge simpson, the simpsons"],
    },
    "soldier_(team_fortress_2)": {
        "character": ["soldier_(team_fortress_2)"],
        "trigger": ["soldier \\(team fortress 2\\), valve"],
    },
    "valoo": {"character": ["valoo"], "trigger": ["valoo, the legend of zelda"]},
    "ramaelfox": {"character": ["ramaelfox"], "trigger": ["ramaelfox, mythology"]},
    "bandaid_protagonist_(tas)": {
        "character": ["bandaid_protagonist_(tas)"],
        "trigger": ["bandaid protagonist \\(tas\\), lifewonders"],
    },
    "james_mccloud": {
        "character": ["james_mccloud"],
        "trigger": ["james mccloud, nintendo"],
    },
    "rita_(disney)": {
        "character": ["rita_(disney)"],
        "trigger": ["rita \\(disney\\), disney"],
    },
    "jaina_proudmoore": {
        "character": ["jaina_proudmoore"],
        "trigger": ["jaina proudmoore, warcraft"],
    },
    "roselyn_(twokinds)": {
        "character": ["roselyn_(twokinds)"],
        "trigger": ["roselyn \\(twokinds\\), twokinds"],
    },
    "bouncyotter": {
        "character": ["bouncyotter"],
        "trigger": ["bouncyotter, mythology"],
    },
    "mirabelle": {"character": ["mirabelle"], "trigger": ["mirabelle, mythology"]},
    "rottytops": {
        "character": ["rottytops"],
        "trigger": ["rottytops, shantae \\(series\\)"],
    },
    "ikshun": {"character": ["ikshun"], "trigger": ["ikshun, mythology"]},
    "assassin_shuten-douji": {
        "character": ["assassin_shuten-douji"],
        "trigger": ["assassin shuten-douji, fate \\(series\\)"],
    },
    "rouen_(shining)": {
        "character": ["rouen_(shining)"],
        "trigger": ["rouen \\(shining\\), sega"],
    },
    "cuddles_(htf)": {
        "character": ["cuddles_(htf)"],
        "trigger": ["cuddles \\(htf\\), happy tree friends"],
    },
    "gang_xi_siyu": {
        "character": ["gang_xi_siyu"],
        "trigger": ["gang xi siyu, mythology"],
    },
    "dtz_(cdrr)": {"character": ["dtz_(cdrr)"], "trigger": ["dtz \\(cdrr\\), disney"]},
    "yellow_(shiro-neko)": {
        "character": ["yellow_(shiro-neko)"],
        "trigger": ["yellow \\(shiro-neko\\), pokemon"],
    },
    "sgt._o'fera_(cuphead)": {
        "character": ["sgt._o'fera_(cuphead)"],
        "trigger": ["sgt. o'fera \\(cuphead\\), cuphead \\(game\\)"],
    },
    "ember_(spyro)": {
        "character": ["ember_(spyro)"],
        "trigger": ["ember \\(spyro\\), spyro the dragon"],
    },
    "build_tiger_(character)": {
        "character": ["build_tiger_(character)"],
        "trigger": ["build tiger \\(character\\), build tiger"],
    },
    "thresh": {"character": ["thresh"], "trigger": ["thresh, riot games"]},
    "kyle_(animal_crossing)": {
        "character": ["kyle_(animal_crossing)"],
        "trigger": ["kyle \\(animal crossing\\), animal crossing"],
    },
    "segway_(segwayrulz)": {
        "character": ["segway_(segwayrulz)"],
        "trigger": ["segway \\(segwayrulz\\), mythology"],
    },
    "violet_hopps": {
        "character": ["violet_hopps"],
        "trigger": ["violet hopps, disney"],
    },
    "hisaki_(live_a_hero)": {
        "character": ["hisaki_(live_a_hero)"],
        "trigger": ["hisaki \\(live a hero\\), lifewonders"],
    },
    "patachu": {"character": ["patachu"], "trigger": ["patachu, digimon"]},
    "pink_panther": {
        "character": ["pink_panther"],
        "trigger": ["pink panther, pink panther \\(series\\)"],
    },
    "horus": {"character": ["horus"], "trigger": ["horus, egyptian mythology"]},
    "giggles_(htf)": {
        "character": ["giggles_(htf)"],
        "trigger": ["giggles \\(htf\\), happy tree friends"],
    },
    "wendy_pleakley": {
        "character": ["wendy_pleakley"],
        "trigger": ["wendy pleakley, disney"],
    },
    "penny_ling": {"character": ["penny_ling"], "trigger": ["penny ling, hasbro"]},
    "lion_(steven_universe)": {
        "character": ["lion_(steven_universe)"],
        "trigger": ["lion \\(steven universe\\), cartoon network"],
    },
    "maeve_(twokinds)": {
        "character": ["maeve_(twokinds)"],
        "trigger": ["maeve \\(twokinds\\), twokinds"],
    },
    "sekk'ral": {"character": ["sekk'ral"], "trigger": ["sekk'ral, mythology"]},
    "star_the_spineless_hedgehog": {
        "character": ["star_the_spineless_hedgehog"],
        "trigger": ["star the spineless hedgehog, sonic the hedgehog \\(series\\)"],
    },
    "rain_(cimarron)": {
        "character": ["rain_(cimarron)"],
        "trigger": ["rain \\(cimarron\\), dreamworks"],
    },
    "spider-gwen": {"character": ["spider-gwen"], "trigger": ["spider-gwen, marvel"]},
    "yossi": {"character": ["yossi"], "trigger": ["yossi, telemonster"]},
    "bodi_(rock_dog)": {
        "character": ["bodi_(rock_dog)"],
        "trigger": ["bodi \\(rock dog\\), rock dog"],
    },
    "plum_(plumbold)": {
        "character": ["plum_(plumbold)"],
        "trigger": ["plum \\(plumbold\\), mythology"],
    },
    "retsuko's_mother": {
        "character": ["retsuko's_mother"],
        "trigger": ["retsuko's mother, sanrio"],
    },
    "zabrina_(afc)": {
        "character": ["zabrina_(afc)"],
        "trigger": ["zabrina \\(afc\\), pokemon"],
    },
    "jill_(james_howard)": {
        "character": ["jill_(james_howard)"],
        "trigger": ["jill \\(james howard\\), patreon"],
    },
    "maliketh_(elden_ring)": {
        "character": ["maliketh_(elden_ring)"],
        "trigger": ["maliketh \\(elden ring\\), fromsoftware"],
    },
    "tramp_(lady_and_the_tramp)": {
        "character": ["tramp_(lady_and_the_tramp)"],
        "trigger": ["tramp \\(lady and the tramp\\), disney"],
    },
    "leroy_(lilo_and_stitch)": {
        "character": ["leroy_(lilo_and_stitch)"],
        "trigger": ["leroy \\(lilo and stitch\\), disney"],
    },
    "penelope_pussycat": {
        "character": ["penelope_pussycat"],
        "trigger": ["penelope pussycat, warner brothers"],
    },
    "superman": {"character": ["superman"], "trigger": ["superman, dc comics"]},
    "roger_rabbit": {
        "character": ["roger_rabbit"],
        "trigger": ["roger rabbit, disney"],
    },
    "leo_(velociripper)": {
        "character": ["leo_(velociripper)"],
        "trigger": ["leo \\(velociripper\\), pokemon"],
    },
    "fox's_sister_(kinokoningen)": {
        "character": ["fox's_sister_(kinokoningen)"],
        "trigger": ["fox's sister \\(kinokoningen\\), sony corporation"],
    },
    "hu_ku_li_(milkytiger1145)": {
        "character": ["hu_ku_li_(milkytiger1145)"],
        "trigger": ["hu ku li \\(milkytiger1145\\), mythology"],
    },
    "fury_bowser": {
        "character": ["fury_bowser"],
        "trigger": ["fury bowser, mario bros"],
    },
    "rammy_lamb": {
        "character": ["rammy_lamb"],
        "trigger": ["rammy lamb, um jammer lammy"],
    },
    "wilykat": {"character": ["wilykat"], "trigger": ["wilykat, thundercats"]},
    "elaine_(furryjibe)": {
        "character": ["elaine_(furryjibe)"],
        "trigger": ["elaine \\(furryjibe\\), mythology"],
    },
    "astrid_hofferson": {
        "character": ["astrid_hofferson"],
        "trigger": ["astrid hofferson, how to train your dragon"],
    },
    "azir_(lol)": {
        "character": ["azir_(lol)"],
        "trigger": ["azir \\(lol\\), riot games"],
    },
    "mother_puss": {
        "character": ["mother_puss"],
        "trigger": ["mother puss, the complex adventures of eddie puss"],
    },
    "jane_doe_(hladilnik)": {
        "character": ["jane_doe_(hladilnik)"],
        "trigger": ["jane doe \\(hladilnik\\), smith & wesson"],
    },
    "dragon_(petruz)": {
        "character": ["dragon_(petruz)"],
        "trigger": ["dragon \\(petruz\\), mythology"],
    },
    "jack-jackal_(character)": {
        "character": ["jack-jackal_(character)"],
        "trigger": ["jack-jackal \\(character\\), christmas"],
    },
    "bea_(pokemon)": {
        "character": ["bea_(pokemon)"],
        "trigger": ["bea \\(pokemon\\), pokemon"],
    },
    "captain_bokko": {
        "character": ["captain_bokko"],
        "trigger": ["captain bokko, the amazing 3"],
    },
    "wolfywetfurr": {
        "character": ["wolfywetfurr"],
        "trigger": ["wolfywetfurr, mythology"],
    },
    "ty_hanson": {"character": ["ty_hanson"], "trigger": ["ty hanson, mythology"]},
    "sylvanas_windrunner": {
        "character": ["sylvanas_windrunner"],
        "trigger": ["sylvanas windrunner, warcraft"],
    },
    "kenny_(kenashcorp)": {
        "character": ["kenny_(kenashcorp)"],
        "trigger": ["kenny \\(kenashcorp\\), nintendo"],
    },
    "susie_mccallister": {
        "character": ["susie_mccallister"],
        "trigger": ["susie mccallister, cartoon network"],
    },
    "tansy_(tansydrawsnsfw)": {
        "character": ["tansy_(tansydrawsnsfw)"],
        "trigger": ["tansy \\(tansydrawsnsfw\\), pokemon"],
    },
    "fempyro": {"character": ["fempyro"], "trigger": ["fempyro, valve"]},
    "malon": {"character": ["malon"], "trigger": ["malon, nintendo"]},
    "hoity_toity_(mlp)": {
        "character": ["hoity_toity_(mlp)"],
        "trigger": ["hoity toity \\(mlp\\), my little pony"],
    },
    "princess_koopa": {
        "character": ["princess_koopa"],
        "trigger": ["princess koopa, mario bros"],
    },
    "rocketgirl": {"character": ["rocketgirl"], "trigger": ["rocketgirl, nintendo"]},
    "dinosaur_(google_chrome)": {
        "character": ["dinosaur_(google_chrome)"],
        "trigger": ["dinosaur \\(google chrome\\), google"],
    },
    "tahm_kench_(lol)": {
        "character": ["tahm_kench_(lol)"],
        "trigger": ["tahm kench \\(lol\\), riot games"],
    },
    "cala_maria": {
        "character": ["cala_maria"],
        "trigger": ["cala maria, cuphead \\(game\\)"],
    },
    "niffty_(hazbin_hotel)": {
        "character": ["niffty_(hazbin_hotel)"],
        "trigger": ["niffty \\(hazbin hotel\\), hazbin hotel"],
    },
    "lana_(pokemon)": {
        "character": ["lana_(pokemon)"],
        "trigger": ["lana \\(pokemon\\), pokemon"],
    },
    "ms._chalice": {
        "character": ["ms._chalice"],
        "trigger": ["ms. chalice, cuphead \\(game\\)"],
    },
    "hail_(7th-r)": {
        "character": ["hail_(7th-r)"],
        "trigger": ["hail \\(7th-r\\), nintendo"],
    },
    "senko-san": {
        "character": ["senko-san"],
        "trigger": ["senko-san, sewayaki kitsune no senko-san"],
    },
    "hammerhead_(petruz)": {
        "character": ["hammerhead_(petruz)"],
        "trigger": ["hammerhead \\(petruz\\), petruz \\(copyright\\)"],
    },
    "doctor_(arknights)": {
        "character": ["doctor_(arknights)"],
        "trigger": ["doctor \\(arknights\\), studio montagne"],
    },
    "inco_(iwhtg)": {
        "character": ["inco_(iwhtg)"],
        "trigger": ["inco \\(iwhtg\\), cavemanon studios"],
    },
    "ambar": {"character": ["ambar"], "trigger": ["ambar, las lindas"]},
    "lori_(jmh)": {
        "character": ["lori_(jmh)"],
        "trigger": ["lori \\(jmh\\), christmas"],
    },
    "leo_(saitama_seibu_lions)": {
        "character": ["leo_(saitama_seibu_lions)"],
        "trigger": ["leo \\(saitama seibu lions\\), nippon professional baseball"],
    },
    "alexandra_(velocitycat)": {
        "character": ["alexandra_(velocitycat)"],
        "trigger": ["alexandra \\(velocitycat\\), patreon"],
    },
    "hajime_tanaka_(odd_taxi)": {
        "character": ["hajime_tanaka_(odd_taxi)"],
        "trigger": ["hajime tanaka \\(odd taxi\\), odd taxi"],
    },
    "sprout_cloverleaf_(mlp)": {
        "character": ["sprout_cloverleaf_(mlp)"],
        "trigger": ["sprout cloverleaf \\(mlp\\), my little pony"],
    },
    "geno_e._vefa_(coyotek)": {
        "character": ["geno_e._vefa_(coyotek)"],
        "trigger": ["geno e. vefa \\(coyotek\\), mythology"],
    },
    "klaus_doberman_(character)": {
        "character": ["klaus_doberman_(character)"],
        "trigger": ["klaus doberman \\(character\\), mythology"],
    },
    "kaeldu": {"character": ["kaeldu"], "trigger": ["kaeldu, mythology"]},
    "kindle_(frisky_ferals)": {
        "character": ["kindle_(frisky_ferals)"],
        "trigger": ["kindle \\(frisky ferals\\), mythology"],
    },
    "rareel": {"character": ["rareel"], "trigger": ["rareel, mythology"]},
    "blackjack_o'hare": {
        "character": ["blackjack_o'hare"],
        "trigger": ["blackjack o'hare, marvel"],
    },
    "maku_(maku450)": {
        "character": ["maku_(maku450)"],
        "trigger": ["maku \\(maku450\\), mythology"],
    },
    "kyuu_(beastars)": {
        "character": ["kyuu_(beastars)"],
        "trigger": ["kyuu \\(beastars\\), beastars"],
    },
    "nihea_avarta": {
        "character": ["nihea_avarta"],
        "trigger": ["nihea avarta, european mythology"],
    },
    "crocodile_(petruz)": {
        "character": ["crocodile_(petruz)"],
        "trigger": ["crocodile \\(petruz\\), petruz \\(copyright\\)"],
    },
    "lady_olivia": {"character": ["lady_olivia"], "trigger": ["lady olivia, disney"]},
    "antoine_d'coolette": {
        "character": ["antoine_d'coolette"],
        "trigger": ["antoine d'coolette, sonic the hedgehog \\(series\\)"],
    },
    "tiffany_valentine": {
        "character": ["tiffany_valentine"],
        "trigger": ["tiffany valentine, mythology"],
    },
    "littlefoot": {"character": ["littlefoot"], "trigger": ["littlefoot, don bluth"]},
    "arbiter_(halo)": {
        "character": ["arbiter_(halo)"],
        "trigger": ["arbiter \\(halo\\), halo \\(series\\)"],
    },
    "starlow": {"character": ["starlow"], "trigger": ["starlow, mario bros"]},
    "nixuelle": {"character": ["nixuelle"], "trigger": ["nixuelle, halloween"]},
    "kom_(komdog)": {
        "character": ["kom_(komdog)"],
        "trigger": ["kom \\(komdog\\), mario bros"],
    },
    "tinval": {"character": ["tinval"], "trigger": ["tinval, halloween"]},
    "dog_boy_(berseepon09)": {
        "character": ["dog_boy_(berseepon09)"],
        "trigger": ["dog boy \\(berseepon09\\), mythology"],
    },
    "evals": {"character": ["evals"], "trigger": ["evals, twokinds"]},
    "smartypants_(mlp)": {
        "character": ["smartypants_(mlp)"],
        "trigger": ["smartypants \\(mlp\\), my little pony"],
    },
    "berry_ranieri": {
        "character": ["berry_ranieri"],
        "trigger": ["berry ranieri, pokemon"],
    },
    "nightmare_bonnie_(fnaf)": {
        "character": ["nightmare_bonnie_(fnaf)"],
        "trigger": ["nightmare bonnie \\(fnaf\\), scottgames"],
    },
    "sue_(peculiart)": {
        "character": ["sue_(peculiart)"],
        "trigger": ["sue \\(peculiart\\), dandy demons"],
    },
    "wintie": {"character": ["wintie"], "trigger": ["wintie, pokemon"]},
    "cappy_(mario)": {
        "character": ["cappy_(mario)"],
        "trigger": ["cappy \\(mario\\), super mario odyssey"],
    },
    "katsuki_bakugou": {
        "character": ["katsuki_bakugou"],
        "trigger": ["katsuki bakugou, my hero academia"],
    },
    "pepper_(puppkittyfan1)": {
        "character": ["pepper_(puppkittyfan1)"],
        "trigger": ["pepper \\(puppkittyfan1\\), meme clothing"],
    },
    "cassie_(dragon_tales)": {
        "character": ["cassie_(dragon_tales)"],
        "trigger": ["cassie \\(dragon tales\\), dragon tales"],
    },
    "saphira": {"character": ["saphira"], "trigger": ["saphira, mythology"]},
    "talash": {"character": ["talash"], "trigger": ["talash, mythology"]},
    "petunia_(htf)": {
        "character": ["petunia_(htf)"],
        "trigger": ["petunia \\(htf\\), happy tree friends"],
    },
    "deadpool": {"character": ["deadpool"], "trigger": ["deadpool, marvel"]},
    "windy_dripper": {
        "character": ["windy_dripper"],
        "trigger": ["windy dripper, my little pony"],
    },
    "pat_(lapatte)": {
        "character": ["pat_(lapatte)"],
        "trigger": ["pat \\(lapatte\\), pokemon"],
    },
    "kumoko": {
        "character": ["kumoko"],
        "trigger": ["kumoko, so i'm a spider so what?"],
    },
    "opal_(jellydoeopal)": {
        "character": ["opal_(jellydoeopal)"],
        "trigger": ["opal \\(jellydoeopal\\), christmas"],
    },
    "joan_whitecat": {
        "character": ["joan_whitecat"],
        "trigger": ["joan whitecat, sick fun"],
    },
    "zeriara_(character)": {
        "character": ["zeriara_(character)"],
        "trigger": ["zeriara \\(character\\), gorillaz"],
    },
    "backy_(mlp)": {
        "character": ["backy_(mlp)"],
        "trigger": ["backy \\(mlp\\), my little pony"],
    },
    "ash_(sing)": {
        "character": ["ash_(sing)"],
        "trigger": ["ash \\(sing\\), illumination entertainment"],
    },
    "dr._t'ana": {
        "character": ["dr._t'ana"],
        "trigger": ["dr. t'ana, star trek lower decks"],
    },
    "vlagg_(vju79)": {
        "character": ["vlagg_(vju79)"],
        "trigger": ["vlagg \\(vju79\\), mythology"],
    },
    "nataliya_(petruz)": {
        "character": ["nataliya_(petruz)"],
        "trigger": ["nataliya \\(petruz\\), petruz \\(copyright\\)"],
    },
    "spamton_g._spamton": {
        "character": ["spamton_g._spamton"],
        "trigger": ["spamton g. spamton, undertale \\(series\\)"],
    },
    'norithics_"nori"_kusemurai': {
        "character": ['norithics_"nori"_kusemurai'],
        "trigger": ['norithics "nori" kusemurai, nintendo'],
    },
    "spike_(eg)": {
        "character": ["spike_(eg)"],
        "trigger": ["spike \\(eg\\), my little pony"],
    },
    "skye_(animal_crossing)": {
        "character": ["skye_(animal_crossing)"],
        "trigger": ["skye \\(animal crossing\\), animal crossing"],
    },
    "xanderblaze": {"character": ["xanderblaze"], "trigger": ["xanderblaze, nintendo"]},
    "fredbear_(fnaf)": {
        "character": ["fredbear_(fnaf)"],
        "trigger": ["fredbear \\(fnaf\\), scottgames"],
    },
    "deke_(ittybittykittytittys)": {
        "character": ["deke_(ittybittykittytittys)"],
        "trigger": ["deke \\(ittybittykittytittys\\), disney"],
    },
    "allie_von_schwarzenbek": {
        "character": ["allie_von_schwarzenbek"],
        "trigger": ["allie von schwarzenbek, nintendo"],
    },
    "gir_(invader_zim)": {
        "character": ["gir_(invader_zim)"],
        "trigger": ["gir \\(invader zim\\), invader zim"],
    },
    "sandra_(roanoak)": {
        "character": ["sandra_(roanoak)"],
        "trigger": ["sandra \\(roanoak\\), patreon"],
    },
    "troubleshoes_(mlp)": {
        "character": ["troubleshoes_(mlp)"],
        "trigger": ["troubleshoes \\(mlp\\), my little pony"],
    },
    "princess_skystar_(mlp)": {
        "character": ["princess_skystar_(mlp)"],
        "trigger": ["princess skystar \\(mlp\\), my little pony"],
    },
    "fenrir_(tas)": {
        "character": ["fenrir_(tas)"],
        "trigger": ["fenrir \\(tas\\), lifewonders"],
    },
    "blueberry_kobold": {
        "character": ["blueberry_kobold"],
        "trigger": ["blueberry kobold, kobold quest"],
    },
    "the_dark_urge_(baldur's_gate)": {
        "character": ["the_dark_urge_(baldur's_gate)"],
        "trigger": ["the dark urge \\(baldur's gate\\), electronic arts"],
    },
    "oswald_the_lucky_rabbit": {
        "character": ["oswald_the_lucky_rabbit"],
        "trigger": ["oswald the lucky rabbit, disney"],
    },
    "kneesocks_daemon": {
        "character": ["kneesocks_daemon"],
        "trigger": ["kneesocks daemon, panty and stocking with garterbelt"],
    },
    "risky_boots": {
        "character": ["risky_boots"],
        "trigger": ["risky boots, wayforward"],
    },
    "paulo_(bcb)": {
        "character": ["paulo_(bcb)"],
        "trigger": ["paulo \\(bcb\\), bittersweet candy bowl"],
    },
    "bulk_biceps_(mlp)": {
        "character": ["bulk_biceps_(mlp)"],
        "trigger": ["bulk biceps \\(mlp\\), my little pony"],
    },
    "sunil_nevla": {"character": ["sunil_nevla"], "trigger": ["sunil nevla, hasbro"]},
    "lusamine_(pokemon)": {
        "character": ["lusamine_(pokemon)"],
        "trigger": ["lusamine \\(pokemon\\), pokemon"],
    },
    "jinu_(character)": {
        "character": ["jinu_(character)"],
        "trigger": ["jinu \\(character\\), nintendo"],
    },
    "devin_(yungyiff)": {
        "character": ["devin_(yungyiff)"],
        "trigger": ["devin \\(yungyiff\\), nintendo"],
    },
    "amity_blight": {
        "character": ["amity_blight"],
        "trigger": ["amity blight, disney"],
    },
    "rayman": {"character": ["rayman"], "trigger": ["rayman, ubisoft"]},
    "molly_cunningham": {
        "character": ["molly_cunningham"],
        "trigger": ["molly cunningham, disney"],
    },
    "maverick_skye": {
        "character": ["maverick_skye"],
        "trigger": ["maverick skye, nintendo"],
    },
    "kuromaru": {"character": ["kuromaru"], "trigger": ["kuromaru, mythology"]},
    "nami_(lol)": {
        "character": ["nami_(lol)"],
        "trigger": ["nami \\(lol\\), riot games"],
    },
    "marika_(teer)": {
        "character": ["marika_(teer)"],
        "trigger": ["marika \\(teer\\), kobold quest"],
    },
    "jenny_(ajdurai)": {
        "character": ["jenny_(ajdurai)"],
        "trigger": ["jenny \\(ajdurai\\), mythology"],
    },
    "replica_(oc)": {
        "character": ["replica_(oc)"],
        "trigger": ["replica \\(oc\\), my little pony"],
    },
    "nila_(cyancapsule)": {
        "character": ["nila_(cyancapsule)"],
        "trigger": ["nila \\(cyancapsule\\), can't see the haters"],
    },
    "parfait_(plaga)": {
        "character": ["parfait_(plaga)"],
        "trigger": ["parfait \\(plaga\\), halloween"],
    },
    "nyarlathotep_(tas)": {
        "character": ["nyarlathotep_(tas)"],
        "trigger": ["nyarlathotep \\(tas\\), lifewonders"],
    },
    "caesar_(peculiart)": {
        "character": ["caesar_(peculiart)"],
        "trigger": ["caesar \\(peculiart\\), dandy demons"],
    },
    "jimmy_(jamearts)": {
        "character": ["jimmy_(jamearts)"],
        "trigger": ["jimmy \\(jamearts\\), mythology"],
    },
    "lacy_(blazethefox)": {
        "character": ["lacy_(blazethefox)"],
        "trigger": ["lacy \\(blazethefox\\), mythology"],
    },
    "yagi_b.": {"character": ["yagi_b."], "trigger": ["yagi b., mythology"]},
    "achuchones_(unicorn_wars)": {
        "character": ["achuchones_(unicorn_wars)"],
        "trigger": ["achuchones \\(unicorn wars\\), unicorn wars"],
    },
    "ceroba_ketsukane": {
        "character": ["ceroba_ketsukane"],
        "trigger": ["ceroba ketsukane, undertale yellow"],
    },
    "gex_the_gecko": {
        "character": ["gex_the_gecko"],
        "trigger": ["gex the gecko, gex \\(series\\)"],
    },
    "yozora": {"character": ["yozora"], "trigger": ["yozora, mythology"]},
    "vega_(artica)": {
        "character": ["vega_(artica)"],
        "trigger": ["vega \\(artica\\), mythology"],
    },
    "jimmy_(faf)": {
        "character": ["jimmy_(faf)"],
        "trigger": ["jimmy \\(faf\\), fafcomics"],
    },
    "aggie": {"character": ["aggie"], "trigger": ["aggie, nintendo"]},
    "lori_meyers": {
        "character": ["lori_meyers"],
        "trigger": ["lori meyers, night in the woods"],
    },
    "twist_(twistcmyk)": {
        "character": ["twist_(twistcmyk)"],
        "trigger": ["twist \\(twistcmyk\\), nintendo"],
    },
    "nico_(bastriw)": {
        "character": ["nico_(bastriw)"],
        "trigger": ["nico \\(bastriw\\), patreon"],
    },
    "bonnie_(lilo_and_stitch)": {
        "character": ["bonnie_(lilo_and_stitch)"],
        "trigger": ["bonnie \\(lilo and stitch\\), disney"],
    },
    "leaf_(pokemon)": {
        "character": ["leaf_(pokemon)"],
        "trigger": ["leaf \\(pokemon\\), pokemon"],
    },
    "markus_(dowantanaccount)": {
        "character": ["markus_(dowantanaccount)"],
        "trigger": ["markus \\(dowantanaccount\\), mythology"],
    },
    "chip_the_wolf": {
        "character": ["chip_the_wolf"],
        "trigger": ["chip the wolf, cookie crisp"],
    },
    "zeta_the_echidna": {
        "character": ["zeta_the_echidna"],
        "trigger": ["zeta the echidna, sonic the hedgehog \\(series\\)"],
    },
    "poison_trail": {
        "character": ["poison_trail"],
        "trigger": ["poison trail, my little pony"],
    },
    "cameron_(skunkdude13)": {
        "character": ["cameron_(skunkdude13)"],
        "trigger": ["cameron \\(skunkdude13\\), my little pony"],
    },
    "appel": {"character": ["appel"], "trigger": ["appel, my little pony"]},
    "hildegard_(fidchellvore)": {
        "character": ["hildegard_(fidchellvore)"],
        "trigger": ["hildegard \\(fidchellvore\\), pokemon"],
    },
    "bayzan": {"character": ["bayzan"], "trigger": ["bayzan, mythology"]},
    "schwarzpelz": {
        "character": ["schwarzpelz"],
        "trigger": ["schwarzpelz, mythology"],
    },
    "ultra_(ultrabondagefairy)": {
        "character": ["ultra_(ultrabondagefairy)"],
        "trigger": ["ultra \\(ultrabondagefairy\\), mythology"],
    },
    "lilly_(vimhomeless)": {
        "character": ["lilly_(vimhomeless)"],
        "trigger": ["lilly \\(vimhomeless\\), ren and stimpy"],
    },
    "toshu": {"character": ["toshu"], "trigger": ["toshu, lifewonders"]},
    "sheriff_of_nottingham": {
        "character": ["sheriff_of_nottingham"],
        "trigger": ["sheriff of nottingham, disney"],
    },
    "monkey_d._luffy": {
        "character": ["monkey_d._luffy"],
        "trigger": ["monkey d. luffy, one piece"],
    },
    "taki_(takikuroi)": {
        "character": ["taki_(takikuroi)"],
        "trigger": ["taki \\(takikuroi\\), mythology"],
    },
    "sabel": {"character": ["sabel"], "trigger": ["sabel, mythology"]},
    "n_(pokemon)": {
        "character": ["n_(pokemon)"],
        "trigger": ["n \\(pokemon\\), pokemon"],
    },
    "she-venom": {"character": ["she-venom"], "trigger": ["she-venom, marvel"]},
    "sona_(lol)": {
        "character": ["sona_(lol)"],
        "trigger": ["sona \\(lol\\), riot games"],
    },
    "cenny": {"character": ["cenny"], "trigger": ["cenny, mythology"]},
    "remmmy": {"character": ["remmmy"], "trigger": ["remmmy, mythology"]},
    "reese_(animal_crossing)": {
        "character": ["reese_(animal_crossing)"],
        "trigger": ["reese \\(animal crossing\\), animal crossing"],
    },
    "cheese_quesadilla": {
        "character": ["cheese_quesadilla"],
        "trigger": ["cheese quesadilla, mythology"],
    },
    "star_butterfly": {
        "character": ["star_butterfly"],
        "trigger": ["star butterfly, disney"],
    },
    "angelina_marie": {
        "character": ["angelina_marie"],
        "trigger": ["angelina marie, mythology"],
    },
    "makara_(tas)": {
        "character": ["makara_(tas)"],
        "trigger": ["makara \\(tas\\), lifewonders"],
    },
    "fuuga": {"character": ["fuuga"], "trigger": ["fuuga, april fools' day"]},
    "adine_(angels_with_scaly_wings)": {
        "character": ["adine_(angels_with_scaly_wings)"],
        "trigger": ["adine \\(angels with scaly wings\\), angels with scaly wings"],
    },
    "rippley_(fortnite)": {
        "character": ["rippley_(fortnite)"],
        "trigger": ["rippley \\(fortnite\\), fortnite"],
    },
    "litho_(stormysparkler)": {
        "character": ["litho_(stormysparkler)"],
        "trigger": ["litho \\(stormysparkler\\), pokemon"],
    },
    "brock_(pokemon)": {
        "character": ["brock_(pokemon)"],
        "trigger": ["brock \\(pokemon\\), pokemon"],
    },
    "ness": {"character": ["ness"], "trigger": ["ness, earthbound \\(series\\)"]},
    "lilith_aensland": {
        "character": ["lilith_aensland"],
        "trigger": ["lilith aensland, darkstalkers"],
    },
    "jin_(jindragowolf)": {
        "character": ["jin_(jindragowolf)"],
        "trigger": ["jin \\(jindragowolf\\), mythology"],
    },
    "tails_doll": {
        "character": ["tails_doll"],
        "trigger": ["tails doll, sonic the hedgehog \\(series\\)"],
    },
    "nyan_cat": {
        "character": ["nyan_cat"],
        "trigger": ["nyan cat, nyan cat \\(copyright\\)"],
    },
    "mastertrucker": {
        "character": ["mastertrucker"],
        "trigger": ["mastertrucker, nintendo"],
    },
    "kyo_(kiasano)": {
        "character": ["kyo_(kiasano)"],
        "trigger": ["kyo \\(kiasano\\), dezo"],
    },
    "officer_fangmeyer": {
        "character": ["officer_fangmeyer"],
        "trigger": ["officer fangmeyer, disney"],
    },
    "toffee_(svtfoe)": {
        "character": ["toffee_(svtfoe)"],
        "trigger": ["toffee \\(svtfoe\\), disney"],
    },
    "marylin_(hladilnik)": {
        "character": ["marylin_(hladilnik)"],
        "trigger": ["marylin \\(hladilnik\\), fallout"],
    },
    "crytrauv": {"character": ["crytrauv"], "trigger": ["crytrauv, mythology"]},
    "nessa_(pokemon)": {
        "character": ["nessa_(pokemon)"],
        "trigger": ["nessa \\(pokemon\\), pokemon"],
    },
    "gorou_(genshin_impact)": {
        "character": ["gorou_(genshin_impact)"],
        "trigger": ["gorou \\(genshin impact\\), mihoyo"],
    },
    "jam_(miu)": {
        "character": ["jam_(miu)"],
        "trigger": ["jam \\(miu\\), clubstripes"],
    },
    "grim_reaper": {
        "character": ["grim_reaper"],
        "trigger": ["grim reaper, loving reaper"],
    },
    "florence_ambrose": {
        "character": ["florence_ambrose"],
        "trigger": ["florence ambrose, freefall \\(webcomic\\)"],
    },
    "panchito_pistoles": {
        "character": ["panchito_pistoles"],
        "trigger": ["panchito pistoles, disney"],
    },
    "opera_kranz": {
        "character": ["opera_kranz"],
        "trigger": ["opera kranz, solatorobo"],
    },
    "she-ra_(she-ra)": {
        "character": ["she-ra_(she-ra)"],
        "trigger": ["she-ra \\(she-ra\\), she-ra \\(copyright\\)"],
    },
    "sebrina_arbok": {
        "character": ["sebrina_arbok"],
        "trigger": ["sebrina arbok, pokemon"],
    },
    "cherry_jubilee_(mlp)": {
        "character": ["cherry_jubilee_(mlp)"],
        "trigger": ["cherry jubilee \\(mlp\\), my little pony"],
    },
    "elizabeth_(bioshock_infinite)": {
        "character": ["elizabeth_(bioshock_infinite)"],
        "trigger": ["elizabeth \\(bioshock infinite\\), bioshock"],
    },
    "clove_the_pronghorn": {
        "character": ["clove_the_pronghorn"],
        "trigger": ["clove the pronghorn, sonic the hedgehog \\(series\\)"],
    },
    "rina_(ratcha)": {
        "character": ["rina_(ratcha)"],
        "trigger": ["rina \\(ratcha\\), nintendo switch"],
    },
    "sophie_(funkybun)": {
        "character": ["sophie_(funkybun)"],
        "trigger": ["sophie \\(funkybun\\), nintendo"],
    },
    "minedoo_(character)": {
        "character": ["minedoo_(character)"],
        "trigger": ["minedoo \\(character\\), mythology"],
    },
    "kensuke_shibagaki_(odd_taxi)": {
        "character": ["kensuke_shibagaki_(odd_taxi)"],
        "trigger": ["kensuke shibagaki \\(odd taxi\\), odd taxi"],
    },
    "pasadena_o'possum": {
        "character": ["pasadena_o'possum"],
        "trigger": ["pasadena o'possum, crash bandicoot \\(series\\)"],
    },
    "eva_(ozawk)": {
        "character": ["eva_(ozawk)"],
        "trigger": ["eva \\(ozawk\\), mythology"],
    },
    "waylon_(thecosmicwolf33)": {
        "character": ["waylon_(thecosmicwolf33)"],
        "trigger": ["waylon \\(thecosmicwolf33\\), mythology"],
    },
    "chase_hunter": {
        "character": ["chase_hunter"],
        "trigger": ["chase hunter, echo \\(game\\)"],
    },
    "noah_(downthehatch)": {
        "character": ["noah_(downthehatch)"],
        "trigger": ["noah \\(downthehatch\\), gamecube"],
    },
    "nugget_(diadorin)": {
        "character": ["nugget_(diadorin)"],
        "trigger": ["nugget \\(diadorin\\), bethesda softworks"],
    },
    "raziel_(caelum_sky)": {
        "character": ["raziel_(caelum_sky)"],
        "trigger": ["raziel \\(caelum sky\\), caelum sky"],
    },
    "sam_(changing_fates)": {
        "character": ["sam_(changing_fates)"],
        "trigger": ["sam \\(changing fates\\), mythology"],
    },
    "inka_(welwraith)": {
        "character": ["inka_(welwraith)"],
        "trigger": ["inka \\(welwraith\\), mythology"],
    },
    "zed_burrows": {"character": ["zed_burrows"], "trigger": ["zed burrows, pokemon"]},
    "glados": {"character": ["glados"], "trigger": ["glados, valve"]},
    "sobek": {"character": ["sobek"], "trigger": ["sobek, mythology"]},
    "sek-raktaa": {"character": ["sek-raktaa"], "trigger": ["sek-raktaa, mythology"]},
    "deeja": {"character": ["deeja"], "trigger": ["deeja, the elder scrolls"]},
    "pearl_(boolean)": {
        "character": ["pearl_(boolean)"],
        "trigger": ["pearl \\(boolean\\), mythology"],
    },
    "caradhina": {"character": ["caradhina"], "trigger": ["caradhina, nintendo"]},
    "dusk_rhine": {
        "character": ["dusk_rhine"],
        "trigger": ["dusk rhine, my little pony"],
    },
    "amelia_steelheart": {
        "character": ["amelia_steelheart"],
        "trigger": ["amelia steelheart, mythology"],
    },
    "bonnie_bovine_(character)": {
        "character": ["bonnie_bovine_(character)"],
        "trigger": ["bonnie bovine \\(character\\), mythology"],
    },
    "jin_macchiato": {
        "character": ["jin_macchiato"],
        "trigger": ["jin macchiato, fuga: melodies of steel"],
    },
    "freya_(animal_crossing)": {
        "character": ["freya_(animal_crossing)"],
        "trigger": ["freya \\(animal crossing\\), animal crossing"],
    },
    "yamaneko_sougi": {
        "character": ["yamaneko_sougi"],
        "trigger": ["yamaneko sougi, chimangetsu"],
    },
    "majin_buu": {"character": ["majin_buu"], "trigger": ["majin buu, dragon ball"]},
    "sylvia_(wander_over_yonder)": {
        "character": ["sylvia_(wander_over_yonder)"],
        "trigger": ["sylvia \\(wander over yonder\\), wander over yonder"],
    },
    "evelia_zara": {
        "character": ["evelia_zara"],
        "trigger": ["evelia zara, mythology"],
    },
    "tamara_fox": {"character": ["tamara_fox"], "trigger": ["tamara fox, halloween"]},
    "humphrey_(canisfidelis)": {
        "character": ["humphrey_(canisfidelis)"],
        "trigger": ["humphrey \\(canisfidelis\\), no nut november"],
    },
    "sophie_slam": {
        "character": ["sophie_slam"],
        "trigger": ["sophie slam, super planet dolan"],
    },
    "rabbid_peach": {
        "character": ["rabbid_peach"],
        "trigger": ["rabbid peach, raving rabbids"],
    },
    "chuuta": {
        "character": ["chuuta"],
        "trigger": ["chuuta, gamba no bouken \\(series\\)"],
    },
    "bethany_(jay_naylor)": {
        "character": ["bethany_(jay_naylor)"],
        "trigger": ["bethany \\(jay naylor\\), day of the dead"],
    },
    "hal_(halbean)": {
        "character": ["hal_(halbean)"],
        "trigger": ["hal \\(halbean\\), mythology"],
    },
    "kiggy": {"character": ["kiggy"], "trigger": ["kiggy, halo \\(series\\)"]},
    "bastion_aduro": {
        "character": ["bastion_aduro"],
        "trigger": ["bastion aduro, dreamkeepers"],
    },
    "miranda_lawson": {
        "character": ["miranda_lawson"],
        "trigger": ["miranda lawson, mass effect"],
    },
    "drake_(zerofox)": {
        "character": ["drake_(zerofox)"],
        "trigger": ["drake \\(zerofox\\), mythology"],
    },
    "neera_li": {"character": ["neera_li"], "trigger": ["neera li, freedom planet"]},
    "pawl_(fuze)": {
        "character": ["pawl_(fuze)"],
        "trigger": ["pawl \\(fuze\\), pokemon"],
    },
    "xolotl_(tas)": {
        "character": ["xolotl_(tas)"],
        "trigger": ["xolotl \\(tas\\), lifewonders"],
    },
    "martha_lorraine": {
        "character": ["martha_lorraine"],
        "trigger": ["martha lorraine, martha speaks"],
    },
    "garnet_(jewelpet)": {
        "character": ["garnet_(jewelpet)"],
        "trigger": ["garnet \\(jewelpet\\), jewelpet"],
    },
    "merveille_million": {
        "character": ["merveille_million"],
        "trigger": ["merveille million, solatorobo"],
    },
    "lei-lani": {"character": ["lei-lani"], "trigger": ["lei-lani, the depths"]},
    "lothar": {"character": ["lothar"], "trigger": ["lothar, mythology"]},
    "silverlay_(estories)": {
        "character": ["silverlay_(estories)"],
        "trigger": ["silverlay \\(estories\\), my little pony"],
    },
    "adult_fink": {
        "character": ["adult_fink"],
        "trigger": ["adult fink, cartoon network"],
    },
    "amber_(fuf)": {
        "character": ["amber_(fuf)"],
        "trigger": ["amber \\(fuf\\), pokemon"],
    },
    "parfait_(yesthisisgoocat)": {
        "character": ["parfait_(yesthisisgoocat)"],
        "trigger": ["parfait \\(yesthisisgoocat\\), nintendo"],
    },
    "sadayoshi": {"character": ["sadayoshi"], "trigger": ["sadayoshi, lifewonders"]},
    "ash's_pikachu": {
        "character": ["ash's_pikachu"],
        "trigger": ["ash's pikachu, pokemon"],
    },
    "cornelius_(odin_sphere)": {
        "character": ["cornelius_(odin_sphere)"],
        "trigger": ["cornelius \\(odin sphere\\), odin sphere"],
    },
    "katsuke_(character)": {
        "character": ["katsuke_(character)"],
        "trigger": ["katsuke \\(character\\), mythology"],
    },
    "firestar_(warriors)": {
        "character": ["firestar_(warriors)"],
        "trigger": ["firestar \\(warriors\\), warriors \\(book series\\)"],
    },
    "sebastian_(kadath)": {
        "character": ["sebastian_(kadath)"],
        "trigger": ["sebastian \\(kadath\\), patreon"],
    },
    "zoe_(nnecgrau)": {
        "character": ["zoe_(nnecgrau)"],
        "trigger": ["zoe \\(nnecgrau\\), mythology"],
    },
    "mia_(talash)": {
        "character": ["mia_(talash)"],
        "trigger": ["mia \\(talash\\), patreon"],
    },
    "taiyo_akari": {
        "character": ["taiyo_akari"],
        "trigger": ["taiyo akari, square enix"],
    },
    "lucas_raymond": {
        "character": ["lucas_raymond"],
        "trigger": ["lucas raymond, twitter"],
    },
    "warfare_toriel": {
        "character": ["warfare_toriel"],
        "trigger": ["warfare toriel, undertale \\(series\\)"],
    },
    "marnie_(pokemon)": {
        "character": ["marnie_(pokemon)"],
        "trigger": ["marnie \\(pokemon\\), pokemon"],
    },
    "paimon_(helluva_boss)": {
        "character": ["paimon_(helluva_boss)"],
        "trigger": ["paimon \\(helluva boss\\), helluva boss"],
    },
    "brown_wantholf": {
        "character": ["brown_wantholf"],
        "trigger": ["brown wantholf, nintendo"],
    },
    "tabra": {"character": ["tabra"], "trigger": ["tabra, mythology"]},
    "fancypants_(mlp)": {
        "character": ["fancypants_(mlp)"],
        "trigger": ["fancypants \\(mlp\\), my little pony"],
    },
    "gausswolf": {"character": ["gausswolf"], "trigger": ["gausswolf, mythology"]},
    "shadow_bonnie_(fnaf)": {
        "character": ["shadow_bonnie_(fnaf)"],
        "trigger": ["shadow bonnie \\(fnaf\\), scottgames"],
    },
    "addison_(frisky_ferals)": {
        "character": ["addison_(frisky_ferals)"],
        "trigger": ["addison \\(frisky ferals\\), frisky ferals"],
    },
    "green_(shiro-neko)": {
        "character": ["green_(shiro-neko)"],
        "trigger": ["green \\(shiro-neko\\), pokemon"],
    },
    "blackjack_(pinkbutterfree)": {
        "character": ["blackjack_(pinkbutterfree)"],
        "trigger": ["blackjack \\(pinkbutterfree\\), nintendo"],
    },
    "cthugha_(tas)": {
        "character": ["cthugha_(tas)"],
        "trigger": ["cthugha \\(tas\\), lifewonders"],
    },
    "natasha_(jmh)": {
        "character": ["natasha_(jmh)"],
        "trigger": ["natasha \\(jmh\\), christmas"],
    },
    "cell_(dragon_ball)": {
        "character": ["cell_(dragon_ball)"],
        "trigger": ["cell \\(dragon ball\\), dragon ball"],
    },
    "female_shepard": {
        "character": ["female_shepard"],
        "trigger": ["female shepard, mass effect"],
    },
    "filbert_(animal_crossing)": {
        "character": ["filbert_(animal_crossing)"],
        "trigger": ["filbert \\(animal crossing\\), animal crossing"],
    },
    "sparkx": {"character": ["sparkx"], "trigger": ["sparkx, nintendo"]},
    "peable": {"character": ["peable"], "trigger": ["peable, mythology"]},
    "avery_(roanoak)": {
        "character": ["avery_(roanoak)"],
        "trigger": ["avery \\(roanoak\\), patreon"],
    },
    "grillby": {
        "character": ["grillby"],
        "trigger": ["grillby, undertale \\(series\\)"],
    },
    "mordecai_(lemondeer)": {
        "character": ["mordecai_(lemondeer)"],
        "trigger": ["mordecai \\(lemondeer\\), mythology"],
    },
    "penny_(anaugi)": {
        "character": ["penny_(anaugi)"],
        "trigger": ["penny \\(anaugi\\), nintendo"],
    },
    "vetra_nyx": {"character": ["vetra_nyx"], "trigger": ["vetra nyx, mass effect"]},
    "kaimstain": {"character": ["kaimstain"], "trigger": ["kaimstain, pokemon"]},
    "kiyoshiro_higashimitarai": {
        "character": ["kiyoshiro_higashimitarai"],
        "trigger": ["kiyoshiro higashimitarai, digimon"],
    },
    "ali_(domibun)": {
        "character": ["ali_(domibun)"],
        "trigger": ["ali \\(domibun\\), warfare machine"],
    },
    "sage_(gvh)": {
        "character": ["sage_(gvh)"],
        "trigger": ["sage \\(gvh\\), goodbye volcano high"],
    },
    "shaorune": {
        "character": ["shaorune"],
        "trigger": ["shaorune, tales of \\(series\\)"],
    },
    "pumbaa": {"character": ["pumbaa"], "trigger": ["pumbaa, disney"]},
    "taffy_(las_lindas)": {
        "character": ["taffy_(las_lindas)"],
        "trigger": ["taffy \\(las lindas\\), las lindas"],
    },
    "margie_(animal_crossing)": {
        "character": ["margie_(animal_crossing)"],
        "trigger": ["margie \\(animal crossing\\), animal crossing"],
    },
    "cyrakhis": {"character": ["cyrakhis"], "trigger": ["cyrakhis, mythology"]},
    "squilliam_fancyson": {
        "character": ["squilliam_fancyson"],
        "trigger": ["squilliam fancyson, spongebob squarepants"],
    },
    "sis_(fyoshi)": {
        "character": ["sis_(fyoshi)"],
        "trigger": ["sis \\(fyoshi\\), pokemon"],
    },
    "justin_(ieaden)": {
        "character": ["justin_(ieaden)"],
        "trigger": ["justin \\(ieaden\\), mythology"],
    },
    "milo_stefferson": {
        "character": ["milo_stefferson"],
        "trigger": ["milo stefferson, patreon"],
    },
    "samantha_(syronck01)": {
        "character": ["samantha_(syronck01)"],
        "trigger": ["samantha \\(syronck01\\), christmas"],
    },
    "djoser_(psychoh13)": {
        "character": ["djoser_(psychoh13)"],
        "trigger": ["djoser \\(psychoh13\\), mythology"],
    },
    "ken_ashcorp": {"character": ["ken_ashcorp"], "trigger": ["ken ashcorp, nintendo"]},
    "daphne_blake": {
        "character": ["daphne_blake"],
        "trigger": ["daphne blake, scooby-doo \\(series\\)"],
    },
    "pongo": {"character": ["pongo"], "trigger": ["pongo, disney"]},
    "cheese_sandwich_(mlp)": {
        "character": ["cheese_sandwich_(mlp)"],
        "trigger": ["cheese sandwich \\(mlp\\), my little pony"],
    },
    "dovahkiin": {
        "character": ["dovahkiin"],
        "trigger": ["dovahkiin, the elder scrolls"],
    },
    "doctor_neo_cortex": {
        "character": ["doctor_neo_cortex"],
        "trigger": ["doctor neo cortex, crash bandicoot \\(series\\)"],
    },
    "mileena": {"character": ["mileena"], "trigger": ["mileena, mortal kombat"]},
    "papi_(monster_musume)": {
        "character": ["papi_(monster_musume)"],
        "trigger": ["papi \\(monster musume\\), monster musume"],
    },
    "bailey_(brogulls)": {
        "character": ["bailey_(brogulls)"],
        "trigger": ["bailey \\(brogulls\\), brogulls"],
    },
    "rubella_the_worgen": {
        "character": ["rubella_the_worgen"],
        "trigger": ["rubella the worgen, warcraft"],
    },
    "raripunk": {"character": ["raripunk"], "trigger": ["raripunk, my little pony"]},
    "blanca_(nicky_illust)": {
        "character": ["blanca_(nicky_illust)"],
        "trigger": ["blanca \\(nicky illust\\), disney"],
    },
    "mittens_(bolt)": {
        "character": ["mittens_(bolt)"],
        "trigger": ["mittens \\(bolt\\), disney"],
    },
    "jar_jar_binks": {
        "character": ["jar_jar_binks"],
        "trigger": ["jar jar binks, star wars"],
    },
    "hamtaro": {"character": ["hamtaro"], "trigger": ["hamtaro, hamtaro \\(series\\)"]},
    "sparky_(lilo_and_stitch)": {
        "character": ["sparky_(lilo_and_stitch)"],
        "trigger": ["sparky \\(lilo and stitch\\), disney"],
    },
    "volk_(wjyw)": {
        "character": ["volk_(wjyw)"],
        "trigger": ["volk \\(wjyw\\), soyuzmultfilm"],
    },
    "vinzin_(character)": {
        "character": ["vinzin_(character)"],
        "trigger": ["vinzin \\(character\\), nintendo"],
    },
    "clarabelle_cow": {
        "character": ["clarabelle_cow"],
        "trigger": ["clarabelle cow, disney"],
    },
    "zal": {"character": ["zal"], "trigger": ["zal, mythology"]},
    "komasan": {"character": ["komasan"], "trigger": ["komasan, yo-kai watch"]},
    "bessi_the_bat": {
        "character": ["bessi_the_bat"],
        "trigger": ["bessi the bat, sonic the hedgehog \\(series\\)"],
    },
    "molly_(angstrom)": {
        "character": ["molly_(angstrom)"],
        "trigger": ["molly \\(angstrom\\), pokemon"],
    },
    "stack_(character)": {
        "character": ["stack_(character)"],
        "trigger": ["stack \\(character\\), stack's womb marking"],
    },
    "edgar_(the_summoning)": {
        "character": ["edgar_(the_summoning)"],
        "trigger": ["edgar \\(the summoning\\), cartoon hangover"],
    },
    "shamrock_(lysergide)": {
        "character": ["shamrock_(lysergide)"],
        "trigger": ["shamrock \\(lysergide\\), pokemon"],
    },
    "boosette": {"character": ["boosette"], "trigger": ["boosette, nintendo"]},
    "jesse_cat": {"character": ["jesse_cat"], "trigger": ["jesse cat, mythology"]},
    "allan_(zourik)": {
        "character": ["allan_(zourik)"],
        "trigger": ["allan \\(zourik\\), pokemon"],
    },
    "vanessa_(fnaf)": {
        "character": ["vanessa_(fnaf)"],
        "trigger": ["vanessa \\(fnaf\\), five nights at freddy's"],
    },
    "cloud_strife": {
        "character": ["cloud_strife"],
        "trigger": ["cloud strife, square enix"],
    },
    "anise_(freckles)": {
        "character": ["anise_(freckles)"],
        "trigger": ["anise \\(freckles\\), my little pony"],
    },
    "kairel": {"character": ["kairel"], "trigger": ["kairel, el arca"]},
    "nidalee_(lol)": {
        "character": ["nidalee_(lol)"],
        "trigger": ["nidalee \\(lol\\), riot games"],
    },
    "savage_(doktor-savage)": {
        "character": ["savage_(doktor-savage)"],
        "trigger": ["savage \\(doktor-savage\\), mythology"],
    },
    "volga_(kemo_coliseum)": {
        "character": ["volga_(kemo_coliseum)"],
        "trigger": ["volga \\(kemo coliseum\\), kemo coliseum"],
    },
    "slimshod": {"character": ["slimshod"], "trigger": ["slimshod, mythology"]},
    "pyrocynical": {
        "character": ["pyrocynical"],
        "trigger": ["pyrocynical, scottgames"],
    },
    "gao_(fuze)": {"character": ["gao_(fuze)"], "trigger": ["gao \\(fuze\\), pokemon"]},
    "leo_alvarez": {
        "character": ["leo_alvarez"],
        "trigger": ["leo alvarez, echo \\(game\\)"],
    },
    "ibuki_(beastars)": {
        "character": ["ibuki_(beastars)"],
        "trigger": ["ibuki \\(beastars\\), beastars"],
    },
    "inugami_korone": {
        "character": ["inugami_korone"],
        "trigger": ["inugami korone, hololive"],
    },
    "bebop": {
        "character": ["bebop"],
        "trigger": ["bebop, teenage mutant ninja turtles"],
    },
    "alicia_acorn": {
        "character": ["alicia_acorn"],
        "trigger": ["alicia acorn, sonic the hedgehog \\(series\\)"],
    },
    "nitobe": {"character": ["nitobe"], "trigger": ["nitobe, tooboe bookmark"]},
    "seaward_skies": {
        "character": ["seaward_skies"],
        "trigger": ["seaward skies, my little pony"],
    },
    "arthur_(furfragged)": {
        "character": ["arthur_(furfragged)"],
        "trigger": ["arthur \\(furfragged\\), mythology"],
    },
    "chief_komiya": {
        "character": ["chief_komiya"],
        "trigger": ["chief komiya, sanrio"],
    },
    "chester_ringtail_magreer": {
        "character": ["chester_ringtail_magreer"],
        "trigger": ["chester ringtail magreer, havoc inc."],
    },
    "soutarou_(morenatsu)": {
        "character": ["soutarou_(morenatsu)"],
        "trigger": ["soutarou \\(morenatsu\\), morenatsu"],
    },
    "peter_griffin": {
        "character": ["peter_griffin"],
        "trigger": ["peter griffin, family guy"],
    },
    "kyra_(greyshores)": {
        "character": ["kyra_(greyshores)"],
        "trigger": ["kyra \\(greyshores\\), mythology"],
    },
    "skippy": {"character": ["skippy"], "trigger": ["skippy, disney"]},
    "garrosh_hellscream": {
        "character": ["garrosh_hellscream"],
        "trigger": ["garrosh hellscream, warcraft"],
    },
    "blathers_(animal_crossing)": {
        "character": ["blathers_(animal_crossing)"],
        "trigger": ["blathers \\(animal crossing\\), animal crossing"],
    },
    "rick_(ratcha)": {
        "character": ["rick_(ratcha)"],
        "trigger": ["rick \\(ratcha\\), nintendo"],
    },
    "yura_kousuke": {
        "character": ["yura_kousuke"],
        "trigger": ["yura kousuke, mythology"],
    },
    "sidra_romani": {
        "character": ["sidra_romani"],
        "trigger": ["sidra romani, pokemon"],
    },
    "da_vinci_(101_dalmatians)": {
        "character": ["da_vinci_(101_dalmatians)"],
        "trigger": ["da vinci \\(101 dalmatians\\), disney"],
    },
    "kris_(zourik)": {
        "character": ["kris_(zourik)"],
        "trigger": ["kris \\(zourik\\), mythology"],
    },
    "pyon_(lewdchuu)": {
        "character": ["pyon_(lewdchuu)"],
        "trigger": ["pyon \\(lewdchuu\\), nintendo"],
    },
    "echoen": {"character": ["echoen"], "trigger": ["echoen, mythology"]},
    "vel_valentine_(strawberrycrux)": {
        "character": ["vel_valentine_(strawberrycrux)"],
        "trigger": ["vel valentine \\(strawberrycrux\\), mythology"],
    },
    "buffalo_bell": {
        "character": ["buffalo_bell"],
        "trigger": ["buffalo bell, orix buffaloes"],
    },
    "rarity_(eg)": {
        "character": ["rarity_(eg)"],
        "trigger": ["rarity \\(eg\\), my little pony"],
    },
    "bundadingy": {"character": ["bundadingy"], "trigger": ["bundadingy, mythology"]},
    "reiko_usagi": {"character": ["reiko_usagi"], "trigger": ["reiko usagi, rascals"]},
    "yello": {"character": ["yello"], "trigger": ["yello, apple inc."]},
    "lion-san": {"character": ["lion-san"], "trigger": ["lion-san, christmas"]},
    "sekhmet_(link2004)": {
        "character": ["sekhmet_(link2004)"],
        "trigger": ["sekhmet \\(link2004\\), mythology"],
    },
    "franchesca_(garasaki)": {
        "character": ["franchesca_(garasaki)"],
        "trigger": ["franchesca \\(garasaki\\), disney"],
    },
    "kobayashi_(dragon_maid)": {
        "character": ["kobayashi_(dragon_maid)"],
        "trigger": ["kobayashi \\(dragon maid\\), miss kobayashi's dragon maid"],
    },
    "wilbur_(animal_crossing)": {
        "character": ["wilbur_(animal_crossing)"],
        "trigger": ["wilbur \\(animal crossing\\), animal crossing"],
    },
    "valentino_(hazbin_hotel)": {
        "character": ["valentino_(hazbin_hotel)"],
        "trigger": ["valentino \\(hazbin hotel\\), hazbin hotel"],
    },
    "sil_blackmon": {
        "character": ["sil_blackmon"],
        "trigger": ["sil blackmon, pokemon"],
    },
    "king_of_sorrow": {
        "character": ["king_of_sorrow"],
        "trigger": ["king of sorrow, bandai namco"],
    },
    "shandi": {"character": ["shandi"], "trigger": ["shandi, mythology"]},
    "jesse_collins": {
        "character": ["jesse_collins"],
        "trigger": ["jesse collins, sexy mad science"],
    },
    "sheila_(spyro)": {
        "character": ["sheila_(spyro)"],
        "trigger": ["sheila \\(spyro\\), spyro the dragon"],
    },
    "georgia_lockheart": {
        "character": ["georgia_lockheart"],
        "trigger": ["georgia lockheart, my little pony"],
    },
    "dark_samus": {"character": ["dark_samus"], "trigger": ["dark samus, nintendo"]},
    "gylala": {"character": ["gylala"], "trigger": ["gylala, blender \\(software\\)"]},
    "golden_brooch": {
        "character": ["golden_brooch"],
        "trigger": ["golden brooch, my little pony"],
    },
    "druid_(bloons)": {
        "character": ["druid_(bloons)"],
        "trigger": ["druid \\(bloons\\), ninja kiwi"],
    },
    "jesus_christ": {
        "character": ["jesus_christ"],
        "trigger": ["jesus christ, mythology"],
    },
    "unico": {"character": ["unico"], "trigger": ["unico, unico \\(series\\)"]},
    "anuv": {"character": ["anuv"], "trigger": ["anuv, mythology"]},
    "maggie_pesky": {
        "character": ["maggie_pesky"],
        "trigger": ["maggie pesky, disney"],
    },
    "bijou_(hamtaro)": {
        "character": ["bijou_(hamtaro)"],
        "trigger": ["bijou \\(hamtaro\\), hamtaro \\(series\\)"],
    },
    "hecarim_(lol)": {
        "character": ["hecarim_(lol)"],
        "trigger": ["hecarim \\(lol\\), riot games"],
    },
    "fiona_maray": {
        "character": ["fiona_maray"],
        "trigger": ["fiona maray, mythology"],
    },
    "keith_(marsminer)": {
        "character": ["keith_(marsminer)"],
        "trigger": ["keith \\(marsminer\\), made in abyss"],
    },
    "collin_(helluva_boss)": {
        "character": ["collin_(helluva_boss)"],
        "trigger": ["collin \\(helluva boss\\), helluva boss"],
    },
    '"honest"_john_foulfellow': {
        "character": ['"honest"_john_foulfellow'],
        "trigger": ['"honest" john foulfellow, pinocchio'],
    },
    "giancarlo_rosato": {
        "character": ["giancarlo_rosato"],
        "trigger": ["giancarlo rosato, digimon"],
    },
    "lexington_(gargoyles)": {
        "character": ["lexington_(gargoyles)"],
        "trigger": ["lexington \\(gargoyles\\), disney"],
    },
    "database_error_(twokinds)": {
        "character": ["database_error_(twokinds)"],
        "trigger": ["database error \\(twokinds\\), twokinds"],
    },
    "e._aster_bunnymund": {
        "character": ["e._aster_bunnymund"],
        "trigger": ["e. aster bunnymund, rise of the guardians"],
    },
    "hanul": {"character": ["hanul"], "trigger": ["hanul, christmas"]},
    "haziq_(hazumazu)": {
        "character": ["haziq_(hazumazu)"],
        "trigger": ["haziq \\(hazumazu\\), pocky"],
    },
    "clive_(doneru)": {
        "character": ["clive_(doneru)"],
        "trigger": ["clive \\(doneru\\), mythology"],
    },
    "amelie_(bunnybits)": {
        "character": ["amelie_(bunnybits)"],
        "trigger": ["amelie \\(bunnybits\\), nintendo"],
    },
    "hazel_weiss": {
        "character": ["hazel_weiss"],
        "trigger": ["hazel weiss, furafterdark"],
    },
    "blaze-lupine_(character)": {
        "character": ["blaze-lupine_(character)"],
        "trigger": ["blaze-lupine \\(character\\), my little pony"],
    },
    "chibisuke": {"character": ["chibisuke"], "trigger": ["chibisuke, dragon drive"]},
    "buchi": {"character": ["buchi"], "trigger": ["buchi, mekko rarekko"]},
    "student_mei_ling": {
        "character": ["student_mei_ling"],
        "trigger": ["student mei ling, kung fu panda"],
    },
    "grovyle_the_thief": {
        "character": ["grovyle_the_thief"],
        "trigger": ["grovyle the thief, pokemon mystery dungeon"],
    },
    "featherweight_(mlp)": {
        "character": ["featherweight_(mlp)"],
        "trigger": ["featherweight \\(mlp\\), my little pony"],
    },
    "zerosuit_fox": {
        "character": ["zerosuit_fox"],
        "trigger": ["zerosuit fox, star fox"],
    },
    "rethex": {"character": ["rethex"], "trigger": ["rethex, pokemon"]},
    "juniper_(freckles)": {
        "character": ["juniper_(freckles)"],
        "trigger": ["juniper \\(freckles\\), mythology"],
    },
    "sassy_saddles_(mlp)": {
        "character": ["sassy_saddles_(mlp)"],
        "trigger": ["sassy saddles \\(mlp\\), my little pony"],
    },
    "ballora_(fnafsl)": {
        "character": ["ballora_(fnafsl)"],
        "trigger": ["ballora \\(fnafsl\\), scottgames"],
    },
    "huggles": {"character": ["huggles"], "trigger": ["huggles, mythology"]},
    "aeril_(helios)": {
        "character": ["aeril_(helios)"],
        "trigger": ["aeril \\(helios\\), blender \\(software\\)"],
    },
    "robin_raccoon": {
        "character": ["robin_raccoon"],
        "trigger": ["robin raccoon, blender \\(software\\)"],
    },
    "the_infection_(hollow_knight)": {
        "character": ["the_infection_(hollow_knight)"],
        "trigger": ["the infection \\(hollow knight\\), team cherry"],
    },
    "selkie_(my_hero_academia)": {
        "character": ["selkie_(my_hero_academia)"],
        "trigger": ["selkie \\(my hero academia\\), my hero academia"],
    },
    "tanya_keys": {
        "character": ["tanya_keys"],
        "trigger": ["tanya keys, cartoon network"],
    },
    "agent_torque": {"character": ["agent_torque"], "trigger": ["agent torque, x-com"]},
    "volo_(pokemon)": {
        "character": ["volo_(pokemon)"],
        "trigger": ["volo \\(pokemon\\), pokemon"],
    },
    "surprise_(pre-g4)": {
        "character": ["surprise_(pre-g4)"],
        "trigger": ["surprise \\(pre-g4\\), my little pony"],
    },
    "cera_(the_land_before_time)": {
        "character": ["cera_(the_land_before_time)"],
        "trigger": ["cera \\(the land before time\\), don bluth"],
    },
    "gin_(blackfox85)": {
        "character": ["gin_(blackfox85)"],
        "trigger": ["gin \\(blackfox85\\), nintendo"],
    },
    "raventhan": {"character": ["raventhan"], "trigger": ["raventhan, mythology"]},
    "makari": {"character": ["makari"], "trigger": ["makari, patreon"]},
    "mars_miner": {
        "character": ["mars_miner"],
        "trigger": ["mars miner, my little pony"],
    },
    "max_(hoodie)": {
        "character": ["max_(hoodie)"],
        "trigger": ["max \\(hoodie\\), warcraft"],
    },
    "renato_manchas": {
        "character": ["renato_manchas"],
        "trigger": ["renato manchas, disney"],
    },
    "tinker_doo": {
        "character": ["tinker_doo"],
        "trigger": ["tinker doo, my little pony"],
    },
    "twstacker_(character)": {
        "character": ["twstacker_(character)"],
        "trigger": ["twstacker \\(character\\), mythology"],
    },
    "nyarai_(furryfight_chronicles)": {
        "character": ["nyarai_(furryfight_chronicles)"],
        "trigger": ["nyarai \\(furryfight chronicles\\), furryfight chronicles"],
    },
    "julia_brain": {
        "character": ["julia_brain"],
        "trigger": ["julia brain, warner brothers"],
    },
    "ink-eyes": {
        "character": ["ink-eyes"],
        "trigger": ["ink-eyes, wizards of the coast"],
    },
    "demoman_(team_fortress_2)": {
        "character": ["demoman_(team_fortress_2)"],
        "trigger": ["demoman \\(team fortress 2\\), valve"],
    },
    "morca_(character)": {
        "character": ["morca_(character)"],
        "trigger": ["morca \\(character\\), mythology"],
    },
    "zazu_(the_lion_king)": {
        "character": ["zazu_(the_lion_king)"],
        "trigger": ["zazu \\(the lion king\\), disney"],
    },
    "roko_(cocolog)": {
        "character": ["roko_(cocolog)"],
        "trigger": ["roko \\(cocolog\\), meme clothing"],
    },
    "luna_(sailor_moon)": {
        "character": ["luna_(sailor_moon)"],
        "trigger": ["luna \\(sailor moon\\), sailor moon \\(series\\)"],
    },
    "estella_(zummeng)": {
        "character": ["estella_(zummeng)"],
        "trigger": ["estella \\(zummeng\\), christmas"],
    },
    "yaehara_makoto": {
        "character": ["yaehara_makoto"],
        "trigger": ["yaehara makoto, mythology"],
    },
    "rosie_(roselynn_meadow)": {
        "character": ["rosie_(roselynn_meadow)"],
        "trigger": ["rosie \\(roselynn meadow\\), mythology"],
    },
    "amanda_(tcitw)": {
        "character": ["amanda_(tcitw)"],
        "trigger": ["amanda \\(tcitw\\), the cabin in the woods \\(arania\\)"],
    },
    "ranni_the_witch": {
        "character": ["ranni_the_witch"],
        "trigger": ["ranni the witch, fromsoftware"],
    },
    "arthur_read": {
        "character": ["arthur_read"],
        "trigger": ["arthur read, arthur \\(series\\)"],
    },
    "mushu_(disney)": {
        "character": ["mushu_(disney)"],
        "trigger": ["mushu \\(disney\\), disney"],
    },
    "e-123_omega": {
        "character": ["e-123_omega"],
        "trigger": ["e-123 omega, sonic the hedgehog \\(series\\)"],
    },
    "hioshiru_(character)": {
        "character": ["hioshiru_(character)"],
        "trigger": ["hioshiru \\(character\\), blender \\(software\\)"],
    },
    "turquoise_(ralek)": {
        "character": ["turquoise_(ralek)"],
        "trigger": ["turquoise \\(ralek\\), my little pony"],
    },
    "pyro29": {"character": ["pyro29"], "trigger": ["pyro29, fahleir"]},
    "devon_(frisky_ferals)": {
        "character": ["devon_(frisky_ferals)"],
        "trigger": ["devon \\(frisky ferals\\), frisky ferals"],
    },
    "pepper_(sketchytoasty)": {
        "character": ["pepper_(sketchytoasty)"],
        "trigger": ["pepper \\(sketchytoasty\\), valve"],
    },
    "forl_(thepatchedragon)": {
        "character": ["forl_(thepatchedragon)"],
        "trigger": ["forl \\(thepatchedragon\\), mythology"],
    },
    "alecrast": {"character": ["alecrast"], "trigger": ["alecrast, mythology"]},
    "broadway_(gargoyles)": {
        "character": ["broadway_(gargoyles)"],
        "trigger": ["broadway \\(gargoyles\\), disney"],
    },
    "pistol_pete": {
        "character": ["pistol_pete"],
        "trigger": ["pistol pete, goof troop"],
    },
    "stomak": {"character": ["stomak"], "trigger": ["stomak, nintendo"]},
    "quiet_(metal_gear)": {
        "character": ["quiet_(metal_gear)"],
        "trigger": ["quiet \\(metal gear\\), metal gear"],
    },
    "sifyro": {"character": ["sifyro"], "trigger": ["sifyro, mythology"]},
    "adra_(tiddles)": {
        "character": ["adra_(tiddles)"],
        "trigger": ["adra \\(tiddles\\), nintendo"],
    },
    "wesley_(suave_senpai)": {
        "character": ["wesley_(suave_senpai)"],
        "trigger": ["wesley \\(suave senpai\\), mythology"],
    },
    "sarana_(knotthere)": {
        "character": ["sarana_(knotthere)"],
        "trigger": ["sarana \\(knotthere\\), nintendo"],
    },
    "perrito_(puss_in_boots)": {
        "character": ["perrito_(puss_in_boots)"],
        "trigger": ["perrito \\(puss in boots\\), puss in boots \\(dreamworks\\)"],
    },
    "ammit": {"character": ["ammit"], "trigger": ["ammit, mythology"]},
    "furmessiah_(character)": {
        "character": ["furmessiah_(character)"],
        "trigger": ["furmessiah \\(character\\), guild wars"],
    },
    "cyan_hijirikawa": {
        "character": ["cyan_hijirikawa"],
        "trigger": ["cyan hijirikawa, show by rock!!"],
    },
    "breke_(tas)": {
        "character": ["breke_(tas)"],
        "trigger": ["breke \\(tas\\), lifewonders"],
    },
    "roxanne_wolf_(mayosplash)": {
        "character": ["roxanne_wolf_(mayosplash)"],
        "trigger": [
            "roxanne wolf \\(mayosplash\\), five nights at freddy's: security breach"
        ],
    },
    "pepe_the_frog": {
        "character": ["pepe_the_frog"],
        "trigger": ["pepe the frog, nintendo"],
    },
    "christopher_thorndyke": {
        "character": ["christopher_thorndyke"],
        "trigger": ["christopher thorndyke, sonic the hedgehog \\(series\\)"],
    },
    "kid_cat_(animal_crossing)": {
        "character": ["kid_cat_(animal_crossing)"],
        "trigger": ["kid cat \\(animal crossing\\), animal crossing"],
    },
    "noke_(delicatessen)": {
        "character": ["noke_(delicatessen)"],
        "trigger": ["noke \\(delicatessen\\), nintendo"],
    },
    "mugger_(my_life_with_fel)": {
        "character": ["mugger_(my_life_with_fel)"],
        "trigger": ["mugger \\(my life with fel\\), my life with fel"],
    },
    "mariah_wolves": {
        "character": ["mariah_wolves"],
        "trigger": ["mariah wolves, my little pony"],
    },
    "acrid": {"character": ["acrid"], "trigger": ["acrid, risk of rain"]},
    "withered_foxy_(fnaf)": {
        "character": ["withered_foxy_(fnaf)"],
        "trigger": ["withered foxy \\(fnaf\\), five nights at freddy's 2"],
    },
    "rakuo": {"character": ["rakuo"], "trigger": ["rakuo, mythology"]},
    "darma_(rock_dog)": {
        "character": ["darma_(rock_dog)"],
        "trigger": ["darma \\(rock dog\\), rock dog"],
    },
    "johnny_(sing)": {
        "character": ["johnny_(sing)"],
        "trigger": ["johnny \\(sing\\), illumination entertainment"],
    },
    "marei": {"character": ["marei"], "trigger": ["marei, penis lineup"]},
    "tracy_porter": {
        "character": ["tracy_porter"],
        "trigger": ["tracy porter, general motors"],
    },
    "nicky_equeen": {
        "character": ["nicky_equeen"],
        "trigger": ["nicky equeen, my little pony"],
    },
    "marilin_(welwraith)": {
        "character": ["marilin_(welwraith)"],
        "trigger": ["marilin \\(welwraith\\), mythology"],
    },
    "kikimora_(the_owl_house)": {
        "character": ["kikimora_(the_owl_house)"],
        "trigger": ["kikimora \\(the owl house\\), disney"],
    },
    "sailor_moon_(character)": {
        "character": ["sailor_moon_(character)"],
        "trigger": ["sailor moon \\(character\\), sailor moon \\(series\\)"],
    },
    "aku_aku": {
        "character": ["aku_aku"],
        "trigger": ["aku aku, crash bandicoot \\(series\\)"],
    },
    "pheeze": {"character": ["pheeze"], "trigger": ["pheeze, mythology"]},
    "angela_(gargoyles)": {
        "character": ["angela_(gargoyles)"],
        "trigger": ["angela \\(gargoyles\\), disney"],
    },
    "gamercat_(character)": {
        "character": ["gamercat_(character)"],
        "trigger": ["gamercat \\(character\\), the gamercat"],
    },
    "spirale_(character)": {
        "character": ["spirale_(character)"],
        "trigger": ["spirale \\(character\\), lucky and chocolate charms"],
    },
    "novus": {"character": ["novus"], "trigger": ["novus, beta behave"]},
    "chloe_shiwulf": {
        "character": ["chloe_shiwulf"],
        "trigger": ["chloe shiwulf, microsoft"],
    },
    "thomas_whitaker": {
        "character": ["thomas_whitaker"],
        "trigger": ["thomas whitaker, pokemon"],
    },
    "rai_(radarn)": {
        "character": ["rai_(radarn)"],
        "trigger": ["rai \\(radarn\\), pokemon"],
    },
    "hel_(shiretsuna)": {
        "character": ["hel_(shiretsuna)"],
        "trigger": ["hel \\(shiretsuna\\), mythology"],
    },
    "avo_(weaver)": {
        "character": ["avo_(weaver)"],
        "trigger": ["avo \\(weaver\\), pack street"],
    },
    "ferris_argyle": {
        "character": ["ferris_argyle"],
        "trigger": ["ferris argyle, re:zero"],
    },
    "duncan_(zeromccall)": {
        "character": ["duncan_(zeromccall)"],
        "trigger": ["duncan \\(zeromccall\\), disney"],
    },
    "gary_goodspeed": {
        "character": ["gary_goodspeed"],
        "trigger": ["gary goodspeed, final space"],
    },
    "ethan_(pokemon)": {
        "character": ["ethan_(pokemon)"],
        "trigger": ["ethan \\(pokemon\\), pokemon"],
    },
    "aeon_calcos": {
        "character": ["aeon_calcos"],
        "trigger": ["aeon calcos, soul calibur"],
    },
    "grunt_(mass_effect)": {
        "character": ["grunt_(mass_effect)"],
        "trigger": ["grunt \\(mass effect\\), mass effect"],
    },
    "king_(housepets!)": {
        "character": ["king_(housepets!)"],
        "trigger": ["king \\(housepets!\\), housepets!"],
    },
    "soto_(freckles)": {
        "character": ["soto_(freckles)"],
        "trigger": ["soto \\(freckles\\), mythology"],
    },
    "fate_valentine": {
        "character": ["fate_valentine"],
        "trigger": ["fate valentine, pokemon"],
    },
    "chel": {"character": ["chel"], "trigger": ["chel, the road to el dorado"]},
    "jay_(sqoon)": {
        "character": ["jay_(sqoon)"],
        "trigger": ["jay \\(sqoon\\), halloween"],
    },
    "saryn_(warframe)": {
        "character": ["saryn_(warframe)"],
        "trigger": ["saryn \\(warframe\\), warframe"],
    },
    "wafu": {"character": ["wafu"], "trigger": ["wafu, pokemon"]},
    "evie_(zummeng)": {
        "character": ["evie_(zummeng)"],
        "trigger": ["evie \\(zummeng\\), christmas"],
    },
    "freya_(zionsangel)": {
        "character": ["freya_(zionsangel)"],
        "trigger": ["freya \\(zionsangel\\), mythology"],
    },
    "domi_(domibun)": {
        "character": ["domi_(domibun)"],
        "trigger": ["domi \\(domibun\\), source filmmaker"],
    },
    "belle_(beauty_and_the_beast)": {
        "character": ["belle_(beauty_and_the_beast)"],
        "trigger": ["belle \\(beauty and the beast\\), beauty and the beast"],
    },
    "vriska_serket": {
        "character": ["vriska_serket"],
        "trigger": ["vriska serket, homestuck"],
    },
    "tank_(mlp)": {
        "character": ["tank_(mlp)"],
        "trigger": ["tank \\(mlp\\), my little pony"],
    },
    "mothman": {"character": ["mothman"], "trigger": ["mothman, mythology"]},
    "lizeron": {"character": ["lizeron"], "trigger": ["lizeron, mythology"]},
    "dubmare": {"character": ["dubmare"], "trigger": ["dubmare, glock"]},
    "ahab_(tas)": {
        "character": ["ahab_(tas)"],
        "trigger": ["ahab \\(tas\\), lifewonders"],
    },
    "hollyhock_(bojack_horseman)": {
        "character": ["hollyhock_(bojack_horseman)"],
        "trigger": ["hollyhock \\(bojack horseman\\), netflix"],
    },
    "vovo": {"character": ["vovo"], "trigger": ["vovo, pokemon"]},
    "inigo_(wooled)": {
        "character": ["inigo_(wooled)"],
        "trigger": ["inigo \\(wooled\\), pokemon mystery dungeon"],
    },
    "cream_(miu)": {
        "character": ["cream_(miu)"],
        "trigger": ["cream \\(miu\\), clubstripes"],
    },
    "fang_(animal_crossing)": {
        "character": ["fang_(animal_crossing)"],
        "trigger": ["fang \\(animal crossing\\), animal crossing"],
    },
    "ray_vanhem": {"character": ["ray_vanhem"], "trigger": ["ray vanhem, mythology"]},
    "garo_(garoshadowscale)": {
        "character": ["garo_(garoshadowscale)"],
        "trigger": ["garo \\(garoshadowscale\\), mythology"],
    },
    "montimer_(yinller)": {
        "character": ["montimer_(yinller)"],
        "trigger": ["montimer \\(yinller\\), angel in the forest"],
    },
    "zax_(zwalexan)": {
        "character": ["zax_(zwalexan)"],
        "trigger": ["zax \\(zwalexan\\), mythology"],
    },
    "kuuko": {"character": ["kuuko"], "trigger": ["kuuko, tale of tails"]},
    "cerberus_(fortnite)": {
        "character": ["cerberus_(fortnite)"],
        "trigger": ["cerberus \\(fortnite\\), fortnite"],
    },
    "hunter_(road_rovers)": {
        "character": ["hunter_(road_rovers)"],
        "trigger": ["hunter \\(road rovers\\), road rovers"],
    },
    "malik": {"character": ["malik"], "trigger": ["malik, mythology"]},
    "gintaro": {"character": ["gintaro"], "trigger": ["gintaro, gingitsune"]},
    "suel": {"character": ["suel"], "trigger": ["suel, warcraft"]},
    "repede": {"character": ["repede"], "trigger": ["repede, tales of \\(series\\)"]},
    "eric_vaughan": {
        "character": ["eric_vaughan"],
        "trigger": ["eric vaughan, twokinds"],
    },
    "marvol": {"character": ["marvol"], "trigger": ["marvol, mythology"]},
    "double_diamond_(mlp)": {
        "character": ["double_diamond_(mlp)"],
        "trigger": ["double diamond \\(mlp\\), my little pony"],
    },
    "serenakty": {"character": ["serenakty"], "trigger": ["serenakty, rutwell forest"]},
    "victory_(texdot)": {
        "character": ["victory_(texdot)"],
        "trigger": ["victory \\(texdot\\), mythology"],
    },
    "murana_wolford_(darkflame-wolf)": {
        "character": ["murana_wolford_(darkflame-wolf)"],
        "trigger": ["murana wolford \\(darkflame-wolf\\), legend of ahya"],
    },
    "khiara_(personalami)": {
        "character": ["khiara_(personalami)"],
        "trigger": ["khiara \\(personalami\\), mythology"],
    },
    "momo_(monarquis)": {
        "character": ["momo_(monarquis)"],
        "trigger": ["momo \\(monarquis\\), halloween"],
    },
    "sasha_la_fleur": {
        "character": ["sasha_la_fleur"],
        "trigger": ["sasha la fleur, don bluth"],
    },
    "iyo": {"character": ["iyo"], "trigger": ["iyo, animal yokocho"]},
    "dulcine": {"character": ["dulcine"], "trigger": ["dulcine, mythology"]},
    "artoria_pendragon": {
        "character": ["artoria_pendragon"],
        "trigger": ["artoria pendragon, type-moon"],
    },
    "the_great_prince_of_the_forest": {
        "character": ["the_great_prince_of_the_forest"],
        "trigger": ["the great prince of the forest, disney"],
    },
    "jinx_(lol)": {
        "character": ["jinx_(lol)"],
        "trigger": ["jinx \\(lol\\), riot games"],
    },
    "felix_reverie": {
        "character": ["felix_reverie"],
        "trigger": ["felix reverie, mythology"],
    },
    "garret_mvahd_(oc)": {
        "character": ["garret_mvahd_(oc)"],
        "trigger": ["garret mvahd \\(oc\\), mythology"],
    },
    "protagonist_(helltaker)": {
        "character": ["protagonist_(helltaker)"],
        "trigger": ["protagonist \\(helltaker\\), helltaker"],
    },
    "boyfriend_(fnf)": {
        "character": ["boyfriend_(fnf)"],
        "trigger": ["boyfriend \\(fnf\\), friday night funkin'"],
    },
    "toofer": {"character": ["toofer"], "trigger": ["toofer, mythology"]},
    "enko_(mrt0ony)": {
        "character": ["enko_(mrt0ony)"],
        "trigger": ["enko \\(mrt0ony\\), nintendo"],
    },
    "kimiko_five-tails": {
        "character": ["kimiko_five-tails"],
        "trigger": ["kimiko five-tails, fortnite"],
    },
    "daniel_toke": {
        "character": ["daniel_toke"],
        "trigger": ["daniel toke, furaffinity"],
    },
    "sunni_gummi": {"character": ["sunni_gummi"], "trigger": ["sunni gummi, disney"]},
    "gustav_(here_there_be_dragons)": {
        "character": ["gustav_(here_there_be_dragons)"],
        "trigger": ["gustav \\(here there be dragons\\), here there be dragons"],
    },
    "s'zira": {"character": ["s'zira"], "trigger": ["s'zira, digimon"]},
    "maria_robotnik": {
        "character": ["maria_robotnik"],
        "trigger": ["maria robotnik, sonic the hedgehog \\(series\\)"],
    },
    "monara": {"character": ["monara"], "trigger": ["monara, warcraft"]},
    "pecan_(animal_crossing)": {
        "character": ["pecan_(animal_crossing)"],
        "trigger": ["pecan \\(animal crossing\\), animal crossing"],
    },
    "dragonchu_(character)": {
        "character": ["dragonchu_(character)"],
        "trigger": ["dragonchu \\(character\\), nintendo"],
    },
    "lord_shen": {"character": ["lord_shen"], "trigger": ["lord shen, kung fu panda"]},
    "dribble_(warioware)": {
        "character": ["dribble_(warioware)"],
        "trigger": ["dribble \\(warioware\\), warioware"],
    },
    "jeremy_fitzgerald": {
        "character": ["jeremy_fitzgerald"],
        "trigger": ["jeremy fitzgerald, five nights at freddy's 2"],
    },
    "red_(redishdragie)": {
        "character": ["red_(redishdragie)"],
        "trigger": ["red \\(redishdragie\\), mythology"],
    },
    "ashnu": {"character": ["ashnu"], "trigger": ["ashnu, mythology"]},
    "marchosias_(tas)": {
        "character": ["marchosias_(tas)"],
        "trigger": ["marchosias \\(tas\\), lifewonders"],
    },
    "clyde_(discord)": {
        "character": ["clyde_(discord)"],
        "trigger": ["clyde \\(discord\\), discord \\(app\\)"],
    },
    "gato_matero_(character)": {
        "character": ["gato_matero_(character)"],
        "trigger": ["gato matero \\(character\\), halloween"],
    },
    "elias_acorn": {
        "character": ["elias_acorn"],
        "trigger": ["elias acorn, sonic the hedgehog \\(series\\)"],
    },
    "jotaro_kujo": {
        "character": ["jotaro_kujo"],
        "trigger": ["jotaro kujo, jojo's bizarre adventure"],
    },
    "saria": {"character": ["saria"], "trigger": ["saria, the legend of zelda"]},
    "sera_(sera)": {
        "character": ["sera_(sera)"],
        "trigger": ["sera \\(sera\\), mythology"],
    },
    "selene_blackcat": {
        "character": ["selene_blackcat"],
        "trigger": ["selene blackcat, nintendo"],
    },
    "coonix": {"character": ["coonix"], "trigger": ["coonix, mythology"]},
    "ryai_(character)": {
        "character": ["ryai_(character)"],
        "trigger": ["ryai \\(character\\), christmas"],
    },
    "bottom_armor_(lefthighkick)": {
        "character": ["bottom_armor_(lefthighkick)"],
        "trigger": ["bottom armor \\(lefthighkick\\), my little pony"],
    },
    "sektiss": {"character": ["sektiss"], "trigger": ["sektiss, my little pony"]},
    "neferu_(adastra)": {
        "character": ["neferu_(adastra)"],
        "trigger": ["neferu \\(adastra\\), adastra \\(series\\)"],
    },
    "side_b": {"character": ["side_b"], "trigger": ["side b, pokemon"]},
    "devourer_(razor_koopa)": {
        "character": ["devourer_(razor_koopa)"],
        "trigger": ["devourer \\(razor koopa\\), mythology"],
    },
    "kat_vance": {"character": ["kat_vance"], "trigger": ["kat vance, sequential art"]},
    "shaggy_rogers": {
        "character": ["shaggy_rogers"],
        "trigger": ["shaggy rogers, scooby-doo \\(series\\)"],
    },
    "jason_voorhees": {
        "character": ["jason_voorhees"],
        "trigger": ["jason voorhees, friday the 13th \\(series\\)"],
    },
    "d.w._read": {
        "character": ["d.w._read"],
        "trigger": ["d.w. read, arthur \\(series\\)"],
    },
    "kagome_higurashi": {
        "character": ["kagome_higurashi"],
        "trigger": ["kagome higurashi, inuyasha"],
    },
    "aniece": {"character": ["aniece"], "trigger": ["aniece, mythology"]},
    "desmond_(ceeb)": {
        "character": ["desmond_(ceeb)"],
        "trigger": ["desmond \\(ceeb\\), mythology"],
    },
    "diana_(animal_crossing)": {
        "character": ["diana_(animal_crossing)"],
        "trigger": ["diana \\(animal crossing\\), animal crossing"],
    },
    "king_felix": {"character": ["king_felix"], "trigger": ["king felix, digimon"]},
    "william_afton_(fnaf)": {
        "character": ["william_afton_(fnaf)"],
        "trigger": ["william afton \\(fnaf\\), scottgames"],
    },
    "nightmare_fredbear_(fnaf)": {
        "character": ["nightmare_fredbear_(fnaf)"],
        "trigger": ["nightmare fredbear \\(fnaf\\), scottgames"],
    },
    "nicecream_man": {
        "character": ["nicecream_man"],
        "trigger": ["nicecream man, undertale \\(series\\)"],
    },
    "buster_moon": {
        "character": ["buster_moon"],
        "trigger": ["buster moon, illumination entertainment"],
    },
    "skoll_(wolf-skoll)": {
        "character": ["skoll_(wolf-skoll)"],
        "trigger": ["skoll \\(wolf-skoll\\), christmas"],
    },
    "vlue_(maynara)": {
        "character": ["vlue_(maynara)"],
        "trigger": ["vlue \\(maynara\\), nintendo"],
    },
    "cat_(petruz)": {
        "character": ["cat_(petruz)"],
        "trigger": ["cat \\(petruz\\), petruz \\(copyright\\)"],
    },
    "jerry_(sing)": {
        "character": ["jerry_(sing)"],
        "trigger": ["jerry \\(sing\\), illumination entertainment"],
    },
    "peaches_(miu)": {
        "character": ["peaches_(miu)"],
        "trigger": ["peaches \\(miu\\), clubstripes"],
    },
    "kermit_the_frog": {
        "character": ["kermit_the_frog"],
        "trigger": ["kermit the frog, muppets"],
    },
    "duck_hunt_duck": {
        "character": ["duck_hunt_duck"],
        "trigger": ["duck hunt duck, duck hunt"],
    },
    "tigger": {
        "character": ["tigger"],
        "trigger": ["tigger, winnie the pooh \\(franchise\\)"],
    },
    "evil_coco": {
        "character": ["evil_coco"],
        "trigger": ["evil coco, crash bandicoot \\(series\\)"],
    },
    "tarot_(housepets!)": {
        "character": ["tarot_(housepets!)"],
        "trigger": ["tarot \\(housepets!\\), housepets!"],
    },
    "king_adelaide": {
        "character": ["king_adelaide"],
        "trigger": ["king adelaide, twokinds"],
    },
    "iron_will_(mlp)": {
        "character": ["iron_will_(mlp)"],
        "trigger": ["iron will \\(mlp\\), my little pony"],
    },
    "natsume_(tooboe_bookmark)": {
        "character": ["natsume_(tooboe_bookmark)"],
        "trigger": ["natsume \\(tooboe bookmark\\), tooboe bookmark"],
    },
    "wander": {"character": ["wander"], "trigger": ["wander, wander over yonder"]},
    "yaojou": {"character": ["yaojou"], "trigger": ["yaojou, mythology"]},
    "icarus_skyhawk": {
        "character": ["icarus_skyhawk"],
        "trigger": ["icarus skyhawk, mythology"],
    },
    "magna_(armello)": {
        "character": ["magna_(armello)"],
        "trigger": ["magna \\(armello\\), armello"],
    },
    "jiggly_juggle_(oc)": {
        "character": ["jiggly_juggle_(oc)"],
        "trigger": ["jiggly juggle \\(oc\\), my little pony"],
    },
    "berry_frost": {
        "character": ["berry_frost"],
        "trigger": ["berry frost, my little pony"],
    },
    "buddy_thunderstruck_(character)": {
        "character": ["buddy_thunderstruck_(character)"],
        "trigger": [
            "buddy thunderstruck \\(character\\), buddy thunderstruck \\(series\\)"
        ],
    },
    "bat_(petruz)": {
        "character": ["bat_(petruz)"],
        "trigger": ["bat \\(petruz\\), petruz \\(copyright\\)"],
    },
    "larry_(yinller)": {
        "character": ["larry_(yinller)"],
        "trigger": ["larry \\(yinller\\), angel in the forest"],
    },
    "luvashi": {"character": ["luvashi"], "trigger": ["luvashi, mythology"]},
    "lappland_(arknights)": {
        "character": ["lappland_(arknights)"],
        "trigger": ["lappland \\(arknights\\), studio montagne"],
    },
    "to-be-named-later": {
        "character": ["to-be-named-later"],
        "trigger": ["to-be-named-later, mythology"],
    },
    "violet_(artca9)": {
        "character": ["violet_(artca9)"],
        "trigger": ["violet \\(artca9\\), jeep"],
    },
    "frankenstein": {
        "character": ["frankenstein"],
        "trigger": ["frankenstein, halloween"],
    },
    "gillpanda_(character)": {
        "character": ["gillpanda_(character)"],
        "trigger": ["gillpanda \\(character\\), mythology"],
    },
    "karen_taverndatter": {
        "character": ["karen_taverndatter"],
        "trigger": ["karen taverndatter, twokinds"],
    },
    "panthy": {"character": ["panthy"], "trigger": ["panthy, el arca"]},
    "pearl_krabs": {
        "character": ["pearl_krabs"],
        "trigger": ["pearl krabs, spongebob squarepants"],
    },
    "aluka_(dragoon86)": {
        "character": ["aluka_(dragoon86)"],
        "trigger": ["aluka \\(dragoon86\\), mythology"],
    },
    "eris_(marefurryfan)": {
        "character": ["eris_(marefurryfan)"],
        "trigger": ["eris \\(marefurryfan\\), dreamworks"],
    },
    "opalescence_(mlp)": {
        "character": ["opalescence_(mlp)"],
        "trigger": ["opalescence \\(mlp\\), my little pony"],
    },
    "adrian_gray": {"character": ["adrian_gray"], "trigger": ["adrian gray, ah club"]},
    "fluttershy_(eg)": {
        "character": ["fluttershy_(eg)"],
        "trigger": ["fluttershy \\(eg\\), my little pony"],
    },
    "bonbon_(animal_crossing)": {
        "character": ["bonbon_(animal_crossing)"],
        "trigger": ["bonbon \\(animal crossing\\), animal crossing"],
    },
    "weaver_bonnie": {
        "character": ["weaver_bonnie"],
        "trigger": ["weaver bonnie, scottgames"],
    },
    "bow_hothoof_(mlp)": {
        "character": ["bow_hothoof_(mlp)"],
        "trigger": ["bow hothoof \\(mlp\\), my little pony"],
    },
    "bepo_(one_piece)": {
        "character": ["bepo_(one_piece)"],
        "trigger": ["bepo \\(one piece\\), one piece"],
    },
    "lute_(zinfyu)": {
        "character": ["lute_(zinfyu)"],
        "trigger": ["lute \\(zinfyu\\), nintendo"],
    },
    "silvia_(peculiart)": {
        "character": ["silvia_(peculiart)"],
        "trigger": ["silvia \\(peculiart\\), dandy demons"],
    },
    "kaji_(karnator)": {
        "character": ["kaji_(karnator)"],
        "trigger": ["kaji \\(karnator\\), mythology"],
    },
    "balloonist_spyro": {
        "character": ["balloonist_spyro"],
        "trigger": ["balloonist spyro, spyro the dragon"],
    },
    "steve_jovonovich": {
        "character": ["steve_jovonovich"],
        "trigger": ["steve jovonovich, dusk acres"],
    },
    "xing_(the_xing1)": {
        "character": ["xing_(the_xing1)"],
        "trigger": ["xing \\(the xing1\\), nintendo"],
    },
    "ryder_(striped_sins)": {
        "character": ["ryder_(striped_sins)"],
        "trigger": ["ryder \\(striped sins\\), striped sins"],
    },
    "diana_digma": {
        "character": ["diana_digma"],
        "trigger": ["diana digma, mythology"],
    },
    "brianna_(kitfox-crimson)": {
        "character": ["brianna_(kitfox-crimson)"],
        "trigger": ["brianna \\(kitfox-crimson\\), in our shadow"],
    },
    "bagi": {
        "character": ["bagi"],
        "trigger": ["bagi, bagi the monster of mighty nature"],
    },
    "lois_griffin": {
        "character": ["lois_griffin"],
        "trigger": ["lois griffin, family guy"],
    },
    "zashy": {"character": ["zashy"], "trigger": ["zashy, mythology"]},
    "bayonetta_(character)": {
        "character": ["bayonetta_(character)"],
        "trigger": ["bayonetta \\(character\\), platinumgames"],
    },
    "rotor_the_walrus": {
        "character": ["rotor_the_walrus"],
        "trigger": ["rotor the walrus, sonic the hedgehog \\(series\\)"],
    },
    "nevarrio": {"character": ["nevarrio"], "trigger": ["nevarrio, mythology"]},
    "fray_(reysi)": {
        "character": ["fray_(reysi)"],
        "trigger": ["fray \\(reysi\\), disney"],
    },
    "percy_(pickles-hyena)": {
        "character": ["percy_(pickles-hyena)"],
        "trigger": ["percy \\(pickles-hyena\\), family tensions"],
    },
    "little_cato": {
        "character": ["little_cato"],
        "trigger": ["little cato, final space"],
    },
    "marco_(adastra)": {
        "character": ["marco_(adastra)"],
        "trigger": ["marco \\(adastra\\), adastra \\(series\\)"],
    },
    "anubii_(quin-nsfw)": {
        "character": ["anubii_(quin-nsfw)"],
        "trigger": ["anubii \\(quin-nsfw\\), snapchat"],
    },
    "lilith_clawthorne": {
        "character": ["lilith_clawthorne"],
        "trigger": ["lilith clawthorne, disney"],
    },
    "aztep_(azzyyeen)": {
        "character": ["aztep_(azzyyeen)"],
        "trigger": ["aztep \\(azzyyeen\\), pokemon"],
    },
    "krystal_(dogzeela)": {
        "character": ["krystal_(dogzeela)"],
        "trigger": ["krystal \\(dogzeela\\), star fox"],
    },
    "clover_(happy_happy_clover)": {
        "character": ["clover_(happy_happy_clover)"],
        "trigger": ["clover \\(happy happy clover\\), happy happy clover"],
    },
    "foghorn_leghorn": {
        "character": ["foghorn_leghorn"],
        "trigger": ["foghorn leghorn, warner brothers"],
    },
    "sapphie_(jewelpet)": {
        "character": ["sapphie_(jewelpet)"],
        "trigger": ["sapphie \\(jewelpet\\), jewelpet"],
    },
    "nessie_(disney)": {
        "character": ["nessie_(disney)"],
        "trigger": ["nessie \\(disney\\), disney"],
    },
    "taro_(liontaro)": {
        "character": ["taro_(liontaro)"],
        "trigger": ["taro \\(liontaro\\), nintendo"],
    },
    "lexxs": {"character": ["lexxs"], "trigger": ["lexxs, warcraft"]},
    "chipp": {"character": ["chipp"], "trigger": ["chipp, nintendo"]},
    "xayah_(lol)": {
        "character": ["xayah_(lol)"],
        "trigger": ["xayah \\(lol\\), riot games"],
    },
    "mond_reyes": {"character": ["mond_reyes"], "trigger": ["mond reyes, texnatsu"]},
    "gremile_'hotshot'": {
        "character": ["gremile_'hotshot'"],
        "trigger": ["gremile 'hotshot', kings of hell"],
    },
    "scp-956": {"character": ["scp-956"], "trigger": ["scp-956, scp foundation"]},
    "rivey_ravenheart": {
        "character": ["rivey_ravenheart"],
        "trigger": ["rivey ravenheart, pokemon"],
    },
    "bobo_(gamba_no_bouken)": {
        "character": ["bobo_(gamba_no_bouken)"],
        "trigger": ["bobo \\(gamba no bouken\\), gamba no bouken \\(series\\)"],
    },
    "wolf_villain_(live_a_hero)": {
        "character": ["wolf_villain_(live_a_hero)"],
        "trigger": ["wolf villain \\(live a hero\\), lifewonders"],
    },
    "lolo_(klonoa)": {
        "character": ["lolo_(klonoa)"],
        "trigger": ["lolo \\(klonoa\\), bandai namco"],
    },
    "nightshade_(dragonofdarkness1992)": {
        "character": ["nightshade_(dragonofdarkness1992)"],
        "trigger": ["nightshade \\(dragonofdarkness1992\\), mythology"],
    },
    "cheetah_(dc)": {
        "character": ["cheetah_(dc)"],
        "trigger": ["cheetah \\(dc\\), dc comics"],
    },
    "lily_(mlp)": {
        "character": ["lily_(mlp)"],
        "trigger": ["lily \\(mlp\\), my little pony"],
    },
    "blossomforth_(mlp)": {
        "character": ["blossomforth_(mlp)"],
        "trigger": ["blossomforth \\(mlp\\), my little pony"],
    },
    "tappei_(morenatsu)": {
        "character": ["tappei_(morenatsu)"],
        "trigger": ["tappei \\(morenatsu\\), morenatsu"],
    },
    "ross_(rossciaco)": {
        "character": ["ross_(rossciaco)"],
        "trigger": ["ross \\(rossciaco\\), lifewonders"],
    },
    "ginger_(iginger)": {
        "character": ["ginger_(iginger)"],
        "trigger": ["ginger \\(iginger\\), patreon"],
    },
    "yang_xiao_long": {
        "character": ["yang_xiao_long"],
        "trigger": ["yang xiao long, rwby"],
    },
    "shin_(mr-shin)": {
        "character": ["shin_(mr-shin)"],
        "trigger": ["shin \\(mr-shin\\), patreon"],
    },
    "momo_(dagasi)": {
        "character": ["momo_(dagasi)"],
        "trigger": ["momo \\(dagasi\\), nintendo"],
    },
    "yukizard_(evov1)": {
        "character": ["yukizard_(evov1)"],
        "trigger": ["yukizard \\(evov1\\), pokemon"],
    },
    "koyanskaya_(fate)": {
        "character": ["koyanskaya_(fate)"],
        "trigger": ["koyanskaya \\(fate\\), type-moon"],
    },
    "yaya_panda": {
        "character": ["yaya_panda"],
        "trigger": ["yaya panda, crash bandicoot \\(series\\)"],
    },
    "texas_(arknights)": {
        "character": ["texas_(arknights)"],
        "trigger": ["texas \\(arknights\\), studio montagne"],
    },
    "okayu_nekomata": {
        "character": ["okayu_nekomata"],
        "trigger": ["okayu nekomata, hololive"],
    },
    "kaya_(knockedoutdragon)": {
        "character": ["kaya_(knockedoutdragon)"],
        "trigger": ["kaya \\(knockedoutdragon\\), mythology"],
    },
    "xia_(cydonia_xia)": {
        "character": ["xia_(cydonia_xia)"],
        "trigger": ["xia \\(cydonia xia\\), mythology"],
    },
    "sha_(twf)": {
        "character": ["sha_(twf)"],
        "trigger": ["sha \\(twf\\), the walten files"],
    },
    "fuyuki_yamamoto_(odd_taxi)": {
        "character": ["fuyuki_yamamoto_(odd_taxi)"],
        "trigger": ["fuyuki yamamoto \\(odd taxi\\), odd taxi"],
    },
    "cammy_white": {"character": ["cammy_white"], "trigger": ["cammy white, capcom"]},
    "br'er_fox": {
        "character": ["br'er_fox"],
        "trigger": ["br'er fox, song of the south"],
    },
    "book_whitener": {
        "character": ["book_whitener"],
        "trigger": ["book whitener, monochrome \\(series\\)"],
    },
    "liam_(liam-kun)": {
        "character": ["liam_(liam-kun)"],
        "trigger": ["liam \\(liam-kun\\), mythology"],
    },
    "magolor": {"character": ["magolor"], "trigger": ["magolor, kirby \\(series\\)"]},
    "phursie": {"character": ["phursie"], "trigger": ["phursie, nintendo"]},
    "yorha_9s": {"character": ["yorha_9s"], "trigger": ["yorha 9s, platinumgames"]},
    "purah": {"character": ["purah"], "trigger": ["purah, nintendo"]},
    "bright_mac_(mlp)": {
        "character": ["bright_mac_(mlp)"],
        "trigger": ["bright mac \\(mlp\\), my little pony"],
    },
    "hachimitsu": {"character": ["hachimitsu"], "trigger": ["hachimitsu, mythology"]},
    "molly_(slightlysimian)": {
        "character": ["molly_(slightlysimian)"],
        "trigger": ["molly \\(slightlysimian\\), mythology"],
    },
    "fluttergoth": {
        "character": ["fluttergoth"],
        "trigger": ["fluttergoth, my little pony"],
    },
    "olive_(rawk_manx)": {
        "character": ["olive_(rawk_manx)"],
        "trigger": ["olive \\(rawk manx\\), christmas"],
    },
    "dante_kinkade": {
        "character": ["dante_kinkade"],
        "trigger": ["dante kinkade, halloween"],
    },
    "agape_(petruz)": {
        "character": ["agape_(petruz)"],
        "trigger": ["agape \\(petruz\\), mythology"],
    },
    "blake_sinclair": {
        "character": ["blake_sinclair"],
        "trigger": ["blake sinclair, if hell had a taste"],
    },
    "jk_(kemokin_mania)": {
        "character": ["jk_(kemokin_mania)"],
        "trigger": ["jk \\(kemokin mania\\), among us"],
    },
    "elephant_mario": {
        "character": ["elephant_mario"],
        "trigger": ["elephant mario, mario bros"],
    },
    "daigo_(character)": {
        "character": ["daigo_(character)"],
        "trigger": ["daigo \\(character\\), whitekitten"],
    },
    "petey_piranha": {
        "character": ["petey_piranha"],
        "trigger": ["petey piranha, mario bros"],
    },
    "daggett_beaver": {
        "character": ["daggett_beaver"],
        "trigger": ["daggett beaver, the angry beavers"],
    },
    "feretta_(character)": {
        "character": ["feretta_(character)"],
        "trigger": ["feretta \\(character\\), tumblr"],
    },
    "shadow_queen": {
        "character": ["shadow_queen"],
        "trigger": ["shadow queen, mario bros"],
    },
    "kulza": {"character": ["kulza"], "trigger": ["kulza, mythology"]},
    "eddie_puss": {
        "character": ["eddie_puss"],
        "trigger": ["eddie puss, the complex adventures of eddie puss"],
    },
    "excella": {
        "character": ["excella"],
        "trigger": ["excella, alien \\(franchise\\)"],
    },
    "absa": {"character": ["absa"], "trigger": ["absa, rivals of aether"]},
    "matt_(scratch21)": {
        "character": ["matt_(scratch21)"],
        "trigger": ["matt \\(scratch21\\), scratch21"],
    },
    "leo_(twitchyanimation)": {
        "character": ["leo_(twitchyanimation)"],
        "trigger": ["leo \\(twitchyanimation\\), source filmmaker"],
    },
    "ralena/ralaku": {
        "character": ["ralena/ralaku"],
        "trigger": ["ralena/ralaku, pokemon"],
    },
    "momo_(doodle_dip)": {
        "character": ["momo_(doodle_dip)"],
        "trigger": ["momo \\(doodle dip\\), halloween"],
    },
    "rynring": {"character": ["rynring"], "trigger": ["rynring, pokemon"]},
    "lin_(helluva_boss)": {
        "character": ["lin_(helluva_boss)"],
        "trigger": ["lin \\(helluva boss\\), helluva boss"],
    },
    "kalypso": {
        "character": ["kalypso"],
        "trigger": ["kalypso, donkey kong \\(series\\)"],
    },
    "nights": {"character": ["nights"], "trigger": ["nights, sega"]},
    "maho-gato": {"character": ["maho-gato"], "trigger": ["maho-gato, mythology"]},
    "sige": {"character": ["sige"], "trigger": ["sige, nintendo"]},
    "queenie_(shoutingisfun)": {
        "character": ["queenie_(shoutingisfun)"],
        "trigger": ["queenie \\(shoutingisfun\\), pokemon"],
    },
    "mavis_dracula": {
        "character": ["mavis_dracula"],
        "trigger": ["mavis dracula, hotel transylvania"],
    },
    "vyktor_dreygo": {
        "character": ["vyktor_dreygo"],
        "trigger": ["vyktor dreygo, the lusty stallion"],
    },
    "melissa_morgan": {
        "character": ["melissa_morgan"],
        "trigger": ["melissa morgan, super planet dolan"],
    },
    "bucky_oryx-antlerson": {
        "character": ["bucky_oryx-antlerson"],
        "trigger": ["bucky oryx-antlerson, disney"],
    },
    "ferox_(feroxdoon)": {
        "character": ["ferox_(feroxdoon)"],
        "trigger": ["ferox \\(feroxdoon\\), nintendo"],
    },
    "ziina": {"character": ["ziina"], "trigger": ["ziina, mythology"]},
    "draixen": {"character": ["draixen"], "trigger": ["draixen, mythology"]},
    "mercy_(goonie-san)": {
        "character": ["mercy_(goonie-san)"],
        "trigger": ["mercy \\(goonie-san\\), nintendo"],
    },
    "leo_the_magician": {
        "character": ["leo_the_magician"],
        "trigger": ["leo the magician, mythology"],
    },
    "homeless_dog": {
        "character": ["homeless_dog"],
        "trigger": ["homeless dog, scp foundation"],
    },
    "minos": {"character": ["minos"], "trigger": ["minos, las lindas"]},
    "elisa_maza_(gargoyles)": {
        "character": ["elisa_maza_(gargoyles)"],
        "trigger": ["elisa maza \\(gargoyles\\), disney"],
    },
    "rajirra": {"character": ["rajirra"], "trigger": ["rajirra, the elder scrolls"]},
    "afevis_(character)": {
        "character": ["afevis_(character)"],
        "trigger": ["afevis \\(character\\), mythology"],
    },
    "bam_(animal_crossing)": {
        "character": ["bam_(animal_crossing)"],
        "trigger": ["bam \\(animal crossing\\), animal crossing"],
    },
    "garnet_(steven_universe)": {
        "character": ["garnet_(steven_universe)"],
        "trigger": ["garnet \\(steven universe\\), cartoon network"],
    },
    "raven_eevee": {"character": ["raven_eevee"], "trigger": ["raven eevee, pokemon"]},
    "zero_one": {"character": ["zero_one"], "trigger": ["zero one, nintendo"]},
    "jock_protagonist_(tas)": {
        "character": ["jock_protagonist_(tas)"],
        "trigger": ["jock protagonist \\(tas\\), lifewonders"],
    },
    "white_heart_(oc)": {
        "character": ["white_heart_(oc)"],
        "trigger": ["white heart \\(oc\\), my little pony"],
    },
    "junipurr": {"character": ["junipurr"], "trigger": ["junipurr, christmas"]},
    "pat_(bluey)": {
        "character": ["pat_(bluey)"],
        "trigger": ["pat \\(bluey\\), bluey \\(series\\)"],
    },
    "dolero": {"character": ["dolero"], "trigger": ["dolero, mythology"]},
    "ambient_crewmate_(among_us)": {
        "character": ["ambient_crewmate_(among_us)"],
        "trigger": ["ambient crewmate \\(among us\\), ambient among us"],
    },
    "fanmade_design_glamrock_bonnie": {
        "character": ["fanmade_design_glamrock_bonnie"],
        "trigger": ["fanmade design glamrock bonnie, scottgames"],
    },
    "kerfus": {"character": ["kerfus"], "trigger": ["kerfus, pudu robotics"]},
    "lei_(20pesos_sopa)": {
        "character": ["lei_(20pesos_sopa)"],
        "trigger": ["lei \\(20pesos sopa\\), mythology"],
    },
    "asuka_langley_soryu": {
        "character": ["asuka_langley_soryu"],
        "trigger": ["asuka langley soryu, neon genesis evangelion"],
    },
    "akamai": {"character": ["akamai"], "trigger": ["akamai, pokemon"]},
    "luna_paws": {"character": ["luna_paws"], "trigger": ["luna paws, cjrfm"]},
    "siyu": {"character": ["siyu"], "trigger": ["siyu, mythology"]},
    "ryuko_matoi": {
        "character": ["ryuko_matoi"],
        "trigger": ["ryuko matoi, kill la kill"],
    },
    "cassie_(foxydude)": {
        "character": ["cassie_(foxydude)"],
        "trigger": ["cassie \\(foxydude\\), nintendo"],
    },
    "leliel": {"character": ["leliel"], "trigger": ["leliel, ankama"]},
    "darwin_(tinydeerguy)": {
        "character": ["darwin_(tinydeerguy)"],
        "trigger": ["darwin \\(tinydeerguy\\), christmas"],
    },
    "vi_(bug_fables)": {
        "character": ["vi_(bug_fables)"],
        "trigger": ["vi \\(bug fables\\), bug fables"],
    },
    "the_conductor_(ahit)": {
        "character": ["the_conductor_(ahit)"],
        "trigger": ["the conductor \\(ahit\\), a hat in time"],
    },
    "noble_(nakasuji)": {
        "character": ["noble_(nakasuji)"],
        "trigger": ["noble \\(nakasuji\\), mythology"],
    },
    "shun_imai_(odd_taxi)": {
        "character": ["shun_imai_(odd_taxi)"],
        "trigger": ["shun imai \\(odd taxi\\), odd taxi"],
    },
    "warfare_rivet": {
        "character": ["warfare_rivet"],
        "trigger": ["warfare rivet, sony interactive entertainment"],
    },
    "satan": {"character": ["satan"], "trigger": ["satan, convent of hell"]},
    "jasmine_(skidd)": {
        "character": ["jasmine_(skidd)"],
        "trigger": ["jasmine \\(skidd\\), uberquest"],
    },
    "mordecai_heller": {
        "character": ["mordecai_heller"],
        "trigger": ["mordecai heller, lackadaisy"],
    },
    "nutty_(htf)": {
        "character": ["nutty_(htf)"],
        "trigger": ["nutty \\(htf\\), happy tree friends"],
    },
    "osira_(legend_of_queen_opala)": {
        "character": ["osira_(legend_of_queen_opala)"],
        "trigger": ["osira \\(legend of queen opala\\), legend of queen opala"],
    },
    "blue_(sebdoggo)": {
        "character": ["blue_(sebdoggo)"],
        "trigger": ["blue \\(sebdoggo\\), nintendo"],
    },
    "jamie_(novaduskpaw)": {
        "character": ["jamie_(novaduskpaw)"],
        "trigger": ["jamie \\(novaduskpaw\\), novaduskpaw"],
    },
    "katty_katswell": {
        "character": ["katty_katswell"],
        "trigger": ["katty katswell, t.u.f.f. puppy"],
    },
    "bella_(gasaraki2007)": {
        "character": ["bella_(gasaraki2007)"],
        "trigger": ["bella \\(gasaraki2007\\), nintendo"],
    },
    "gab_shiba": {
        "character": ["gab_shiba"],
        "trigger": ["gab shiba, gab \\(comic\\)"],
    },
    "scratazon_leader": {
        "character": ["scratazon_leader"],
        "trigger": ["scratazon leader, ice age \\(series\\)"],
    },
    "tusk_(bleats)": {
        "character": ["tusk_(bleats)"],
        "trigger": ["tusk \\(bleats\\), mythology"],
    },
    "dagger_(sdorica_sunset)": {
        "character": ["dagger_(sdorica_sunset)"],
        "trigger": ["dagger \\(sdorica sunset\\), sdorica"],
    },
    "ahnik_(character)": {
        "character": ["ahnik_(character)"],
        "trigger": ["ahnik \\(character\\), nintendo"],
    },
    "miguno_(beastars)": {
        "character": ["miguno_(beastars)"],
        "trigger": ["miguno \\(beastars\\), beastars"],
    },
    "tashi_gibson": {
        "character": ["tashi_gibson"],
        "trigger": ["tashi gibson, ultimate mating league"],
    },
    "hunter_(rain_world)": {
        "character": ["hunter_(rain_world)"],
        "trigger": ["hunter \\(rain world\\), videocult"],
    },
    "elyssa_(trinity-fate62)": {
        "character": ["elyssa_(trinity-fate62)"],
        "trigger": ["elyssa \\(trinity-fate62\\), mythology"],
    },
    "margaret_(vetisx)": {
        "character": ["margaret_(vetisx)"],
        "trigger": ["margaret \\(vetisx\\), meme clothing"],
    },
    "daryl_vecat": {
        "character": ["daryl_vecat"],
        "trigger": ["daryl vecat, mythology"],
    },
    "sabuke_(character)": {
        "character": ["sabuke_(character)"],
        "trigger": ["sabuke \\(character\\), mythology"],
    },
    "julie_(jhenightfox)": {
        "character": ["julie_(jhenightfox)"],
        "trigger": ["julie \\(jhenightfox\\), pokemon"],
    },
    "basil_(mikrogoat)": {
        "character": ["basil_(mikrogoat)"],
        "trigger": ["basil \\(mikrogoat\\), karen \\(meme\\)"],
    },
    "gammamon_(ghost_game)": {
        "character": ["gammamon_(ghost_game)"],
        "trigger": ["gammamon \\(ghost game\\), digimon"],
    },
    "sarah_kerrigan": {
        "character": ["sarah_kerrigan"],
        "trigger": ["sarah kerrigan, starcraft"],
    },
    "mrs._hudson": {
        "character": ["mrs._hudson"],
        "trigger": ["mrs. hudson, sherlock hound \\(series\\)"],
    },
    "flinters_(character)": {
        "character": ["flinters_(character)"],
        "trigger": ["flinters \\(character\\), jak and daxter"],
    },
    "scamp_(lady_and_the_tramp)": {
        "character": ["scamp_(lady_and_the_tramp)"],
        "trigger": ["scamp \\(lady and the tramp\\), lady and the tramp"],
    },
    "cadpig": {"character": ["cadpig"], "trigger": ["cadpig, disney"]},
    "the_joker": {"character": ["the_joker"], "trigger": ["the joker, dc comics"]},
    "indi_marrallang": {
        "character": ["indi_marrallang"],
        "trigger": ["indi marrallang, dreamkeepers"],
    },
    "sonic.exe": {
        "character": ["sonic.exe"],
        "trigger": ["sonic.exe, sonic the hedgehog \\(series\\)"],
    },
    "zoop": {
        "character": ["zoop"],
        "trigger": ["zoop, sonic the hedgehog \\(series\\)"],
    },
    "hyndrim": {"character": ["hyndrim"], "trigger": ["hyndrim, thefuraticalgamer"]},
    "patch_(ask-patch)": {
        "character": ["patch_(ask-patch)"],
        "trigger": ["patch \\(ask-patch\\), mythology"],
    },
    "taji_amatsukaze": {
        "character": ["taji_amatsukaze"],
        "trigger": ["taji amatsukaze, mythology"],
    },
    "campfire_(buttocher)": {
        "character": ["campfire_(buttocher)"],
        "trigger": ["campfire \\(buttocher\\), mythology"],
    },
    "tharkis": {"character": ["tharkis"], "trigger": ["tharkis, mythology"]},
    "mikey_(mikey6193)": {
        "character": ["mikey_(mikey6193)"],
        "trigger": ["mikey \\(mikey6193\\), nintendo"],
    },
    "chabett": {"character": ["chabett"], "trigger": ["chabett, mythology"]},
    "plump_(character)": {
        "character": ["plump_(character)"],
        "trigger": ["plump \\(character\\), pokemon"],
    },
    "long_(wish_dragon)": {
        "character": ["long_(wish_dragon)"],
        "trigger": ["long \\(wish dragon\\), wish dragon"],
    },
    "prince_borgon": {
        "character": ["prince_borgon"],
        "trigger": ["prince borgon, mythology"],
    },
    "forrest_(chump)": {
        "character": ["forrest_(chump)"],
        "trigger": ["forrest \\(chump\\), fortnite"],
    },
    "astarion_(baldur's_gate)": {
        "character": ["astarion_(baldur's_gate)"],
        "trigger": ["astarion \\(baldur's gate\\), electronic arts"],
    },
    "rave_raccoon": {
        "character": ["rave_raccoon"],
        "trigger": ["rave raccoon, soyuzmultfilm"],
    },
    "skunkhase": {"character": ["skunkhase"], "trigger": ["skunkhase, nintendo"]},
    "whitney_(pokemon)": {
        "character": ["whitney_(pokemon)"],
        "trigger": ["whitney \\(pokemon\\), pokemon"],
    },
    "mordekaiser_(lol)": {
        "character": ["mordekaiser_(lol)"],
        "trigger": ["mordekaiser \\(lol\\), riot games"],
    },
    "caroo_(character)": {
        "character": ["caroo_(character)"],
        "trigger": ["caroo \\(character\\), nintendo"],
    },
    "metal_(character)": {
        "character": ["metal_(character)"],
        "trigger": ["metal \\(character\\), mythology"],
    },
    "ciri": {"character": ["ciri"], "trigger": ["ciri, the witcher"]},
    "nightmare_freddy_(fnaf)": {
        "character": ["nightmare_freddy_(fnaf)"],
        "trigger": ["nightmare freddy \\(fnaf\\), scottgames"],
    },
    "lilotte": {"character": ["lilotte"], "trigger": ["lilotte, ankama"]},
    "theo_hightower": {
        "character": ["theo_hightower"],
        "trigger": ["theo hightower, patreon"],
    },
    "kalie": {"character": ["kalie"], "trigger": ["kalie, mythology"]},
    "fumikage_tokoyami": {
        "character": ["fumikage_tokoyami"],
        "trigger": ["fumikage tokoyami, my hero academia"],
    },
    "ferlo": {"character": ["ferlo"], "trigger": ["ferlo, disney"]},
    "bradley_(stylusknight)": {
        "character": ["bradley_(stylusknight)"],
        "trigger": ["bradley \\(stylusknight\\), nintendo"],
    },
    "blake_rothenberg": {
        "character": ["blake_rothenberg"],
        "trigger": ["blake rothenberg, pokemon"],
    },
    "kelsey_sienna": {
        "character": ["kelsey_sienna"],
        "trigger": ["kelsey sienna, jewish mythology"],
    },
    "freddie_(gundam_build_divers_re:rise)": {
        "character": ["freddie_(gundam_build_divers_re:rise)"],
        "trigger": [
            "freddie \\(gundam build divers re:rise\\), gundam build divers re:rise"
        ],
    },
    "justin_(study_partners)": {
        "character": ["justin_(study_partners)"],
        "trigger": ["justin \\(study partners\\), study partners"],
    },
    "juna_(batspid2)": {
        "character": ["juna_(batspid2)"],
        "trigger": ["juna \\(batspid2\\), european mythology"],
    },
    "babe_bunyan_(tas)": {
        "character": ["babe_bunyan_(tas)"],
        "trigger": ["babe bunyan \\(tas\\), lifewonders"],
    },
    "ra'zim": {"character": ["ra'zim"], "trigger": ["ra'zim, zgf gaming"]},
    "argiopa": {"character": ["argiopa"], "trigger": ["argiopa, warcraft"]},
    "evangelyne_(wakfu)": {
        "character": ["evangelyne_(wakfu)"],
        "trigger": ["evangelyne \\(wakfu\\), ankama"],
    },
    "quetzalcoatl": {
        "character": ["quetzalcoatl"],
        "trigger": ["quetzalcoatl, mesoamerican mythology"],
    },
    "zach_snowfox": {
        "character": ["zach_snowfox"],
        "trigger": ["zach snowfox, mythology"],
    },
    "zack_(thezackrabbit)": {
        "character": ["zack_(thezackrabbit)"],
        "trigger": ["zack \\(thezackrabbit\\), mythology"],
    },
    "jibanyan": {"character": ["jibanyan"], "trigger": ["jibanyan, yo-kai watch"]},
    "rabbit_shopkeeper": {
        "character": ["rabbit_shopkeeper"],
        "trigger": ["rabbit shopkeeper, undertale \\(series\\)"],
    },
    "azurebolt": {"character": ["azurebolt"], "trigger": ["azurebolt, novaduskpaw"]},
    "spica_(aoino)": {
        "character": ["spica_(aoino)"],
        "trigger": ["spica \\(aoino\\), pocky"],
    },
    "davey_(diadorin)": {
        "character": ["davey_(diadorin)"],
        "trigger": ["davey \\(diadorin\\), warhammer \\(franchise\\)"],
    },
    "enid_(ok_k.o.!_lbh)": {
        "character": ["enid_(ok_k.o.!_lbh)"],
        "trigger": ["enid \\(ok k.o.! lbh\\), cartoon network"],
    },
    "daniel_porter": {
        "character": ["daniel_porter"],
        "trigger": ["daniel porter, nascar"],
    },
    "jenni_(jennibutt)": {
        "character": ["jenni_(jennibutt)"],
        "trigger": ["jenni \\(jennibutt\\), christmas"],
    },
    "william_adler": {
        "character": ["william_adler"],
        "trigger": ["william adler, the smoke room"],
    },
    "joss_(funkybun)": {
        "character": ["joss_(funkybun)"],
        "trigger": ["joss \\(funkybun\\), patreon"],
    },
    "sandpancake": {"character": ["sandpancake"], "trigger": ["sandpancake, pokemon"]},
    "unnamed_fox_(utterangle)": {
        "character": ["unnamed_fox_(utterangle)"],
        "trigger": ["unnamed fox \\(utterangle\\), meme clothing"],
    },
    "john_(photolol.03)": {
        "character": ["john_(photolol.03)"],
        "trigger": ["john \\(photolol.03\\), mythology"],
    },
    "hida": {"character": ["hida"], "trigger": ["hida, mythology"]},
    "totoro": {"character": ["totoro"], "trigger": ["totoro, ghibli"]},
    "scratch_(adventures_of_sonic_the_hedgehog)": {
        "character": ["scratch_(adventures_of_sonic_the_hedgehog)"],
        "trigger": [
            "scratch \\(adventures of sonic the hedgehog\\), sonic the hedgehog \\(series\\)"
        ],
    },
    "kaoru_(kitaness)": {
        "character": ["kaoru_(kitaness)"],
        "trigger": ["kaoru \\(kitaness\\), ohio heat"],
    },
    "princess_vaxi": {
        "character": ["princess_vaxi"],
        "trigger": ["princess vaxi, prince vaxis \\(copyright\\)"],
    },
    "barnaby_kane": {
        "character": ["barnaby_kane"],
        "trigger": ["barnaby kane, mythology"],
    },
    "lantha": {"character": ["lantha"], "trigger": ["lantha, pokemon"]},
    "nogard_krad_nox": {
        "character": ["nogard_krad_nox"],
        "trigger": ["nogard krad nox, mythology"],
    },
    "hekar": {"character": ["hekar"], "trigger": ["hekar, mythology"]},
    "gaster_blaster": {
        "character": ["gaster_blaster"],
        "trigger": ["gaster blaster, undertale \\(series\\)"],
    },
    "tush_(character)": {
        "character": ["tush_(character)"],
        "trigger": ["tush \\(character\\), digimon"],
    },
    "bryce_daeless": {
        "character": ["bryce_daeless"],
        "trigger": ["bryce daeless, pokemon"],
    },
    "helmed_(helmed)": {
        "character": ["helmed_(helmed)"],
        "trigger": ["helmed \\(helmed\\), mythology"],
    },
    "arai-san": {"character": ["arai-san"], "trigger": ["arai-san, kemono friends"]},
    "hyena_father_(pickles-hyena)": {
        "character": ["hyena_father_(pickles-hyena)"],
        "trigger": ["hyena father \\(pickles-hyena\\), family tensions"],
    },
    "seraphine_(roflfox)": {
        "character": ["seraphine_(roflfox)"],
        "trigger": ["seraphine \\(roflfox\\), pokemon"],
    },
    "taylor_knight": {
        "character": ["taylor_knight"],
        "trigger": ["taylor knight, new year"],
    },
    "gabriel_serealia": {
        "character": ["gabriel_serealia"],
        "trigger": ["gabriel serealia, mythology"],
    },
    "starrffax_(fox_sona)": {
        "character": ["starrffax_(fox_sona)"],
        "trigger": ["starrffax \\(fox sona\\), undertale \\(series\\)"],
    },
    "denzel_t_smith_(character)": {
        "character": ["denzel_t_smith_(character)"],
        "trigger": ["denzel t smith \\(character\\), alvin and the chipmunks"],
    },
    "goro_(live_a_hero)": {
        "character": ["goro_(live_a_hero)"],
        "trigger": ["goro \\(live a hero\\), lifewonders"],
    },
    "dijon_(guncht)": {
        "character": ["dijon_(guncht)"],
        "trigger": ["dijon \\(guncht\\), nintendo"],
    },
    "garu_(nu:_carnival)": {
        "character": ["garu_(nu:_carnival)"],
        "trigger": ["garu \\(nu: carnival\\), nu: carnival"],
    },
    "vanilla_(sincastermon)": {
        "character": ["vanilla_(sincastermon)"],
        "trigger": ["vanilla \\(sincastermon\\), ninja kiwi"],
    },
    "calvin_(calvin_and_hobbes)": {
        "character": ["calvin_(calvin_and_hobbes)"],
        "trigger": ["calvin \\(calvin and hobbes\\), calvin and hobbes"],
    },
    "kraft_lawrence": {
        "character": ["kraft_lawrence"],
        "trigger": ["kraft lawrence, spice and wolf"],
    },
    "syldria": {"character": ["syldria"], "trigger": ["syldria, mythology"]},
    "gondar_the_bounty_hunter": {
        "character": ["gondar_the_bounty_hunter"],
        "trigger": ["gondar the bounty hunter, dota"],
    },
    "ollie_(pop'n_music)": {
        "character": ["ollie_(pop'n_music)"],
        "trigger": ["ollie \\(pop'n music\\), pop'n music"],
    },
    "blade_wolf": {"character": ["blade_wolf"], "trigger": ["blade wolf, metal gear"]},
    "youngster_(pokemon)": {
        "character": ["youngster_(pokemon)"],
        "trigger": ["youngster \\(pokemon\\), pokemon"],
    },
    "roadhog_(overwatch)": {
        "character": ["roadhog_(overwatch)"],
        "trigger": ["roadhog \\(overwatch\\), overwatch"],
    },
    "king_shark": {"character": ["king_shark"], "trigger": ["king shark, dc comics"]},
    "cheong_hwan": {
        "character": ["cheong_hwan"],
        "trigger": ["cheong hwan, halloween"],
    },
    "lazuli_(doggod.va)": {
        "character": ["lazuli_(doggod.va)"],
        "trigger": ["lazuli \\(doggod.va\\), nintendo"],
    },
    "caspar_the_frog": {
        "character": ["caspar_the_frog"],
        "trigger": ["caspar the frog, mythology"],
    },
    "warfare_zeraora": {
        "character": ["warfare_zeraora"],
        "trigger": ["warfare zeraora, pokemon"],
    },
    "cotorita": {"character": ["cotorita"], "trigger": ["cotorita, christmas"]},
    "coalt": {"character": ["coalt"], "trigger": ["coalt, mythology"]},
    "cadbury_bunny": {
        "character": ["cadbury_bunny"],
        "trigger": ["cadbury bunny, cadbury"],
    },
    "yamano_taishou": {
        "character": ["yamano_taishou"],
        "trigger": ["yamano taishou, mekko rarekko"],
    },
    "korra": {"character": ["korra"], "trigger": ["korra, the legend of korra"]},
    "night_guard_(mlp)": {
        "character": ["night_guard_(mlp)"],
        "trigger": ["night guard \\(mlp\\), my little pony"],
    },
    "auntie_vixen": {
        "character": ["auntie_vixen"],
        "trigger": ["auntie vixen, metro-goldwyn-mayer"],
    },
    "gabe_(mytigertail)": {
        "character": ["gabe_(mytigertail)"],
        "trigger": ["gabe \\(mytigertail\\), valve"],
    },
    "jayjay_(zoophobia)": {
        "character": ["jayjay_(zoophobia)"],
        "trigger": ["jayjay \\(zoophobia\\), zoophobia"],
    },
    "artemis_the_absol": {
        "character": ["artemis_the_absol"],
        "trigger": ["artemis the absol, pokemon"],
    },
    "laefa_padlo": {"character": ["laefa_padlo"], "trigger": ["laefa padlo, game boy"]},
    "ryn_purrawri": {
        "character": ["ryn_purrawri"],
        "trigger": ["ryn purrawri, pokemon"],
    },
    "summer_(jwinkz)": {
        "character": ["summer_(jwinkz)"],
        "trigger": ["summer \\(jwinkz\\), christmas"],
    },
    "boomer_(nanoff)": {
        "character": ["boomer_(nanoff)"],
        "trigger": ["boomer \\(nanoff\\), nintendo"],
    },
    "mother_kate_(jakethegoat)": {
        "character": ["mother_kate_(jakethegoat)"],
        "trigger": ["mother kate \\(jakethegoat\\), mythology"],
    },
    "hyu": {"character": ["hyu"], "trigger": ["hyu, mythology"]},
    "lucas_(sssonic2)": {
        "character": ["lucas_(sssonic2)"],
        "trigger": ["lucas \\(sssonic2\\), egyptian mythology"],
    },
    "saint_(rain_world)": {
        "character": ["saint_(rain_world)"],
        "trigger": ["saint \\(rain world\\), videocult"],
    },
    "jasmine_(pokemon)": {
        "character": ["jasmine_(pokemon)"],
        "trigger": ["jasmine \\(pokemon\\), pokemon"],
    },
    "faline": {"character": ["faline"], "trigger": ["faline, disney"]},
    "banzai_(the_lion_king)": {
        "character": ["banzai_(the_lion_king)"],
        "trigger": ["banzai \\(the lion king\\), disney"],
    },
    "hunter_(spyro)": {
        "character": ["hunter_(spyro)"],
        "trigger": ["hunter \\(spyro\\), activision"],
    },
    "neferpitou": {
        "character": ["neferpitou"],
        "trigger": ["neferpitou, hunter x hunter"],
    },
    "deiser": {"character": ["deiser"], "trigger": ["deiser, nintendo"]},
    "atoh_darkscythe": {
        "character": ["atoh_darkscythe"],
        "trigger": ["atoh darkscythe, pokemon"],
    },
    "cinnamon_(cinnamoroll)": {
        "character": ["cinnamon_(cinnamoroll)"],
        "trigger": ["cinnamon \\(cinnamoroll\\), cinnamoroll"],
    },
    "amber_steel": {
        "character": ["amber_steel"],
        "trigger": ["amber steel, mythology"],
    },
    "maurice_(nexus)": {
        "character": ["maurice_(nexus)"],
        "trigger": ["maurice \\(nexus\\), universal studios"],
    },
    "lucy_(hladilnik)": {
        "character": ["lucy_(hladilnik)"],
        "trigger": ["lucy \\(hladilnik\\), bible"],
    },
    "hiona": {"character": ["hiona"], "trigger": ["hiona, mythology"]},
    "astolfo_(fate)": {
        "character": ["astolfo_(fate)"],
        "trigger": ["astolfo \\(fate\\), type-moon"],
    },
    "ony": {"character": ["ony"], "trigger": ["ony, nintendo"]},
    "jake_(study_partners)": {
        "character": ["jake_(study_partners)"],
        "trigger": ["jake \\(study partners\\), study partners"],
    },
    "satina": {
        "character": ["satina"],
        "trigger": ["satina, satina wants a glass of water"],
    },
    "keenie_(helluva_boss)": {
        "character": ["keenie_(helluva_boss)"],
        "trigger": ["keenie \\(helluva boss\\), helluva boss"],
    },
    "angoramon_(ghost_game)": {
        "character": ["angoramon_(ghost_game)"],
        "trigger": ["angoramon \\(ghost game\\), digimon"],
    },
    "mommy_long_legs": {
        "character": ["mommy_long_legs"],
        "trigger": ["mommy long legs, poppy playtime"],
    },
    "mage_(final_fantasy)": {
        "character": ["mage_(final_fantasy)"],
        "trigger": ["mage \\(final fantasy\\), square enix"],
    },
    "aerith_gainsborough": {
        "character": ["aerith_gainsborough"],
        "trigger": ["aerith gainsborough, final fantasy vii"],
    },
    "november": {"character": ["november"], "trigger": ["november, mythology"]},
    "rj_(over_the_hedge)": {
        "character": ["rj_(over_the_hedge)"],
        "trigger": ["rj \\(over the hedge\\), over the hedge"],
    },
    "kenai_(brother_bear)": {
        "character": ["kenai_(brother_bear)"],
        "trigger": ["kenai \\(brother bear\\), disney"],
    },
    "vexus": {"character": ["vexus"], "trigger": ["vexus, my life as a teenage robot"]},
    "storm_the_albatross": {
        "character": ["storm_the_albatross"],
        "trigger": ["storm the albatross, sonic the hedgehog \\(series\\)"],
    },
    "rodan_(toho)": {
        "character": ["rodan_(toho)"],
        "trigger": ["rodan \\(toho\\), toho"],
    },
    "uperior": {"character": ["uperior"], "trigger": ["uperior, mythology"]},
    "chani_(ajdurai)": {
        "character": ["chani_(ajdurai)"],
        "trigger": ["chani \\(ajdurai\\), mythology"],
    },
    "kiddy_(todeskiddy)": {
        "character": ["kiddy_(todeskiddy)"],
        "trigger": ["kiddy \\(todeskiddy\\), nintendo"],
    },
    "gary_(zootopia)": {
        "character": ["gary_(zootopia)"],
        "trigger": ["gary \\(zootopia\\), disney"],
    },
    "snavel": {"character": ["snavel"], "trigger": ["snavel, mythology"]},
    "bolero_delatante": {
        "character": ["bolero_delatante"],
        "trigger": ["bolero delatante, pokemon"],
    },
    "hammond_(overwatch)": {
        "character": ["hammond_(overwatch)"],
        "trigger": ["hammond \\(overwatch\\), blizzard entertainment"],
    },
    "ronya": {"character": ["ronya"], "trigger": ["ronya, patreon"]},
    "orville_(animal_crossing)": {
        "character": ["orville_(animal_crossing)"],
        "trigger": ["orville \\(animal crossing\\), animal crossing"],
    },
    "squealers_chief": {
        "character": ["squealers_chief"],
        "trigger": ["squealers chief, brok the investigator"],
    },
    "sepfy": {"character": ["sepfy"], "trigger": ["sepfy, nintendo"]},
    "aku": {"character": ["aku"], "trigger": ["aku, samurai jack"]},
    "tairu": {"character": ["tairu"], "trigger": ["tairu, nintendo"]},
    "wu_sisters": {
        "character": ["wu_sisters"],
        "trigger": ["wu sisters, kung fu panda"],
    },
    "shira_(ice_age)": {
        "character": ["shira_(ice_age)"],
        "trigger": ["shira \\(ice age\\), ice age \\(series\\)"],
    },
    "aiushtha_the_enchantress": {
        "character": ["aiushtha_the_enchantress"],
        "trigger": ["aiushtha the enchantress, dota"],
    },
    "tre": {"character": ["tre"], "trigger": ["tre, nintendo"]},
    "durg": {"character": ["durg"], "trigger": ["durg, mythology"]},
    "oksara_(character)": {
        "character": ["oksara_(character)"],
        "trigger": ["oksara \\(character\\), mythology"],
    },
    "robby_bunny": {"character": ["robby_bunny"], "trigger": ["robby bunny, nintendo"]},
    "aijou": {"character": ["aijou"], "trigger": ["aijou, mythology"]},
    "fraye": {"character": ["fraye"], "trigger": ["fraye, vantanifraye"]},
    "axel_the_tepig": {
        "character": ["axel_the_tepig"],
        "trigger": ["axel the tepig, nintendo"],
    },
    "marc_(smar7)": {
        "character": ["marc_(smar7)"],
        "trigger": ["marc \\(smar7\\), pokemon"],
    },
    "jack_(tcitw)": {
        "character": ["jack_(tcitw)"],
        "trigger": ["jack \\(tcitw\\), the cabin in the woods \\(arania\\)"],
    },
    "holly_zanzibar": {
        "character": ["holly_zanzibar"],
        "trigger": ["holly zanzibar, warfare machine"],
    },
    "chester_the_otter": {
        "character": ["chester_the_otter"],
        "trigger": ["chester the otter, vtuber"],
    },
    "aki_(wingedwilly)": {
        "character": ["aki_(wingedwilly)"],
        "trigger": ["aki \\(wingedwilly\\), pokemon"],
    },
    "robin_(dc)": {
        "character": ["robin_(dc)"],
        "trigger": ["robin \\(dc\\), dc comics"],
    },
    "molly_(koyote)": {
        "character": ["molly_(koyote)"],
        "trigger": ["molly \\(koyote\\), pokemon"],
    },
    "lion-o": {"character": ["lion-o"], "trigger": ["lion-o, thundercats"]},
    "anivia_(lol)": {
        "character": ["anivia_(lol)"],
        "trigger": ["anivia \\(lol\\), riot games"],
    },
    "vem": {"character": ["vem"], "trigger": ["vem, warcraft"]},
    "angela-45": {"character": ["angela-45"], "trigger": ["angela-45, mythology"]},
    "digo_marrallang": {
        "character": ["digo_marrallang"],
        "trigger": ["digo marrallang, dreamkeepers"],
    },
    "muffy_(animal_crossing)": {
        "character": ["muffy_(animal_crossing)"],
        "trigger": ["muffy \\(animal crossing\\), animal crossing"],
    },
    "spitz_(warioware)": {
        "character": ["spitz_(warioware)"],
        "trigger": ["spitz \\(warioware\\), warioware"],
    },
    "tiesci": {"character": ["tiesci"], "trigger": ["tiesci, mythology"]},
    "artemis_tsukino": {
        "character": ["artemis_tsukino"],
        "trigger": ["artemis tsukino, christmas"],
    },
    "hymn_(aogami)": {
        "character": ["hymn_(aogami)"],
        "trigger": ["hymn \\(aogami\\), chirmaya"],
    },
    "ms._zard": {"character": ["ms._zard"], "trigger": ["ms. zard, pokemon"]},
    "febii": {"character": ["febii"], "trigger": ["febii, mythology"]},
    "kel": {"character": ["kel"], "trigger": ["kel, mythology"]},
    "boss_(gym_pals)": {
        "character": ["boss_(gym_pals)"],
        "trigger": ["boss \\(gym pals\\), gym pals"],
    },
    "levin_(levinluxio)": {
        "character": ["levin_(levinluxio)"],
        "trigger": ["levin \\(levinluxio\\), pokemon"],
    },
    "dave_(tcitw)": {
        "character": ["dave_(tcitw)"],
        "trigger": ["dave \\(tcitw\\), the cabin in the woods \\(arania\\)"],
    },
    "faeki_(character)": {
        "character": ["faeki_(character)"],
        "trigger": ["faeki \\(character\\), mythology"],
    },
    "panko_(lawyerdog)": {
        "character": ["panko_(lawyerdog)"],
        "trigger": ["panko \\(lawyerdog\\), christmas"],
    },
    "oli_(thepatchedragon)": {
        "character": ["oli_(thepatchedragon)"],
        "trigger": ["oli \\(thepatchedragon\\), dragonscape"],
    },
    "scarlet_(sequential_art)": {
        "character": ["scarlet_(sequential_art)"],
        "trigger": ["scarlet \\(sequential art\\), sequential art"],
    },
    "copper_(tfath)": {
        "character": ["copper_(tfath)"],
        "trigger": ["copper \\(tfath\\), disney"],
    },
    "cheshire_cat": {
        "character": ["cheshire_cat"],
        "trigger": ["cheshire cat, alice in wonderland"],
    },
    "lunamew": {"character": ["lunamew"], "trigger": ["lunamew, pokemon"]},
    "linda_wright": {
        "character": ["linda_wright"],
        "trigger": ["linda wright, nintendo"],
    },
    "ruby_(rq)": {"character": ["ruby_(rq)"], "trigger": ["ruby \\(rq\\), ruby quest"]},
    "tammy_squirrel": {
        "character": ["tammy_squirrel"],
        "trigger": ["tammy squirrel, disney"],
    },
    "loree": {"character": ["loree"], "trigger": ["loree, halloween"]},
    "backdraft": {"character": ["backdraft"], "trigger": ["backdraft, mythology"]},
    "bonkers_d._bobcat": {
        "character": ["bonkers_d._bobcat"],
        "trigger": ["bonkers d. bobcat, bonkers \\(series\\)"],
    },
    "fatigue_(bedfellows)": {
        "character": ["fatigue_(bedfellows)"],
        "trigger": ["fatigue \\(bedfellows\\), bedfellows"],
    },
    "master_shifu": {
        "character": ["master_shifu"],
        "trigger": ["master shifu, kung fu panda"],
    },
    "pinkie_pie_(eg)": {
        "character": ["pinkie_pie_(eg)"],
        "trigger": ["pinkie pie \\(eg\\), my little pony"],
    },
    "ld": {"character": ["ld"], "trigger": ["ld, my little pony"]},
    "groot": {"character": ["groot"], "trigger": ["groot, guardians of the galaxy"]},
    "yoisho": {
        "character": ["yoisho"],
        "trigger": ["yoisho, gamba no bouken \\(series\\)"],
    },
    "vyrn": {"character": ["vyrn"], "trigger": ["vyrn, cygames"]},
    "zenocoyote_(oc)": {
        "character": ["zenocoyote_(oc)"],
        "trigger": ["zenocoyote \\(oc\\), mythology"],
    },
    "officer_wolfard": {
        "character": ["officer_wolfard"],
        "trigger": ["officer wolfard, disney"],
    },
    "kayla_kitsune": {
        "character": ["kayla_kitsune"],
        "trigger": ["kayla kitsune, the neon city"],
    },
    "pawalo": {"character": ["pawalo"], "trigger": ["pawalo, christmas"]},
    "coal_(samt517)": {
        "character": ["coal_(samt517)"],
        "trigger": ["coal \\(samt517\\), mythology"],
    },
    "grimm_(hollow_knight)": {
        "character": ["grimm_(hollow_knight)"],
        "trigger": ["grimm \\(hollow knight\\), team cherry"],
    },
    "gambit_the_scrafty": {
        "character": ["gambit_the_scrafty"],
        "trigger": ["gambit the scrafty, pokemon"],
    },
    "warfare_amy": {
        "character": ["warfare_amy"],
        "trigger": ["warfare amy, sonic the hedgehog \\(series\\)"],
    },
    "rivulet_(rain_world)": {
        "character": ["rivulet_(rain_world)"],
        "trigger": ["rivulet \\(rain world\\), videocult"],
    },
    "kuya_(nu:_carnival)": {
        "character": ["kuya_(nu:_carnival)"],
        "trigger": ["kuya \\(nu: carnival\\), nu: carnival"],
    },
    "partran_(tiger)": {
        "character": ["partran_(tiger)"],
        "trigger": ["partran \\(tiger\\), winterrock \\(partran\\)"],
    },
    "john_blacksad": {
        "character": ["john_blacksad"],
        "trigger": ["john blacksad, blacksad"],
    },
    "captain_falcon": {
        "character": ["captain_falcon"],
        "trigger": ["captain falcon, f-zero"],
    },
    "panashe_(summon_night)": {
        "character": ["panashe_(summon_night)"],
        "trigger": ["panashe \\(summon night\\), bandai namco"],
    },
    "thea_stilton": {
        "character": ["thea_stilton"],
        "trigger": ["thea stilton, geronimo stilton \\(series\\)"],
    },
    "zeitgeist": {"character": ["zeitgeist"], "trigger": ["zeitgeist, casidhevixen"]},
    "leon_(haychel)": {
        "character": ["leon_(haychel)"],
        "trigger": ["leon \\(haychel\\), pokemon"],
    },
    "applebottom_family": {
        "character": ["applebottom_family"],
        "trigger": ["applebottom family, hollandworks"],
    },
    "mixi_elkhound": {
        "character": ["mixi_elkhound"],
        "trigger": ["mixi elkhound, in-and-awoo"],
    },
    "oxynard": {"character": ["oxynard"], "trigger": ["oxynard, nintendo"]},
    "eglan": {"character": ["eglan"], "trigger": ["eglan, mythology"]},
    "silvia_(pullmytail)": {
        "character": ["silvia_(pullmytail)"],
        "trigger": ["silvia \\(pullmytail\\), mythology"],
    },
    "yukigatr_(evov1)": {
        "character": ["yukigatr_(evov1)"],
        "trigger": ["yukigatr \\(evov1\\), pokemon"],
    },
    "stephie_(fraydia1)": {
        "character": ["stephie_(fraydia1)"],
        "trigger": ["stephie \\(fraydia1\\), mythology"],
    },
    "queen_(deltarune)": {
        "character": ["queen_(deltarune)"],
        "trigger": ["queen \\(deltarune\\), undertale \\(series\\)"],
    },
    "erin_(snoot_game)": {
        "character": ["erin_(snoot_game)"],
        "trigger": ["erin \\(snoot game\\), cavemanon studios"],
    },
    "juliana_(pokemon)": {
        "character": ["juliana_(pokemon)"],
        "trigger": ["juliana \\(pokemon\\), pokemon"],
    },
    "roast_(kumalino)": {
        "character": ["roast_(kumalino)"],
        "trigger": ["roast \\(kumalino\\), kumalino"],
    },
    "eiden_(nu:_carnival)": {
        "character": ["eiden_(nu:_carnival)"],
        "trigger": ["eiden \\(nu: carnival\\), nu: carnival"],
    },
    "rexer": {"character": ["rexer"], "trigger": ["rexer, lifewonders"]},
    "reimu_hakurei": {
        "character": ["reimu_hakurei"],
        "trigger": ["reimu hakurei, touhou"],
    },
    "orion": {"character": ["orion"], "trigger": ["orion, mythology"]},
    "ty_(tygerdenoir)": {
        "character": ["ty_(tygerdenoir)"],
        "trigger": ["ty \\(tygerdenoir\\), mythology"],
    },
    "rainbow_dash_(eg)": {
        "character": ["rainbow_dash_(eg)"],
        "trigger": ["rainbow dash \\(eg\\), my little pony"],
    },
    "wolfgang_(animal_crossing)": {
        "character": ["wolfgang_(animal_crossing)"],
        "trigger": ["wolfgang \\(animal crossing\\), animal crossing"],
    },
    "karma_faye": {"character": ["karma_faye"], "trigger": ["karma faye, patreon"]},
    "birdo_(character)": {
        "character": ["birdo_(character)"],
        "trigger": ["birdo \\(character\\), mario bros"],
    },
    "geo_(pechallai)": {
        "character": ["geo_(pechallai)"],
        "trigger": ["geo \\(pechallai\\), mythology"],
    },
    "sweetie_(paw_patrol)": {
        "character": ["sweetie_(paw_patrol)"],
        "trigger": ["sweetie \\(paw patrol\\), paw patrol"],
    },
    "snow_(matthewdragonblaze)": {
        "character": ["snow_(matthewdragonblaze)"],
        "trigger": ["snow \\(matthewdragonblaze\\), mythology"],
    },
    "huepow": {"character": ["huepow"], "trigger": ["huepow, bandai namco"]},
    "boshi": {
        "character": ["boshi"],
        "trigger": ["boshi, super mario rpg legend of the seven stars"],
    },
    "ashfur_(warriors)": {
        "character": ["ashfur_(warriors)"],
        "trigger": ["ashfur \\(warriors\\), warriors \\(book series\\)"],
    },
    "sheppermint": {
        "character": ["sheppermint"],
        "trigger": ["sheppermint, sailor moon \\(series\\)"],
    },
    "hildegard_rothschild": {
        "character": ["hildegard_rothschild"],
        "trigger": ["hildegard rothschild, ah club"],
    },
    "cyrus_(animal_crossing)": {
        "character": ["cyrus_(animal_crossing)"],
        "trigger": ["cyrus \\(animal crossing\\), animal crossing"],
    },
    "twilight_scepter_(mlp)": {
        "character": ["twilight_scepter_(mlp)"],
        "trigger": ["twilight scepter \\(mlp\\), my little pony"],
    },
    "maxine_d'lapin": {
        "character": ["maxine_d'lapin"],
        "trigger": ["maxine d'lapin, disney"],
    },
    "will_delrio_(sketchybug)": {
        "character": ["will_delrio_(sketchybug)"],
        "trigger": ["will delrio \\(sketchybug\\), monster hunter"],
    },
    "frumples_(character)": {
        "character": ["frumples_(character)"],
        "trigger": ["frumples \\(character\\), patreon"],
    },
    "jonesy_hoovus_(grimart)": {
        "character": ["jonesy_hoovus_(grimart)"],
        "trigger": ["jonesy hoovus \\(grimart\\), mcdonald's"],
    },
    "sonia_(pokemon)": {
        "character": ["sonia_(pokemon)"],
        "trigger": ["sonia \\(pokemon\\), pokemon"],
    },
    "shirota_(aggretsuko)": {
        "character": ["shirota_(aggretsuko)"],
        "trigger": ["shirota \\(aggretsuko\\), sanrio"],
    },
    "keith_(funkybun)": {
        "character": ["keith_(funkybun)"],
        "trigger": ["keith \\(funkybun\\), patreon"],
    },
    "fel_(fenrir)": {
        "character": ["fel_(fenrir)"],
        "trigger": [
            "fel \\(fenrir\\), campfire cooking in another world with my absurd skill"
        ],
    },
    "waffle_ryebread": {
        "character": ["waffle_ryebread"],
        "trigger": ["waffle ryebread, little tail bronx"],
    },
    "guntz": {"character": ["guntz"], "trigger": ["guntz, bandai namco"]},
    "samara": {"character": ["samara"], "trigger": ["samara, mass effect"]},
    "mallow_(happy_happy_clover)": {
        "character": ["mallow_(happy_happy_clover)"],
        "trigger": ["mallow \\(happy happy clover\\), happy happy clover"],
    },
    "olimar": {"character": ["olimar"], "trigger": ["olimar, pikmin"]},
    "whiro": {"character": ["whiro"], "trigger": ["whiro, mythology"]},
    "tanith": {"character": ["tanith"], "trigger": ["tanith, mythology"]},
    "twerp_(halbean)": {
        "character": ["twerp_(halbean)"],
        "trigger": ["twerp \\(halbean\\), mythology"],
    },
    "tech_e._coyote": {
        "character": ["tech_e._coyote"],
        "trigger": ["tech e. coyote, loonatics unleashed"],
    },
    "sandalf": {"character": ["sandalf"], "trigger": ["sandalf, mythology"]},
    "evelynn_(lol)": {
        "character": ["evelynn_(lol)"],
        "trigger": ["evelynn \\(lol\\), riot games"],
    },
    "owlowiscious_(mlp)": {
        "character": ["owlowiscious_(mlp)"],
        "trigger": ["owlowiscious \\(mlp\\), my little pony"],
    },
    "indigo_marrallang": {
        "character": ["indigo_marrallang"],
        "trigger": ["indigo marrallang, dreamkeepers"],
    },
    "mira_(target_miss)": {
        "character": ["mira_(target_miss)"],
        "trigger": ["mira \\(target miss\\), target miss"],
    },
    "aria_(aogami)": {
        "character": ["aria_(aogami)"],
        "trigger": ["aria \\(aogami\\), chirmaya"],
    },
    "nightmare_chica_(fnaf)": {
        "character": ["nightmare_chica_(fnaf)"],
        "trigger": ["nightmare chica \\(fnaf\\), scottgames"],
    },
    "sally_(scalie_schoolie)": {
        "character": ["sally_(scalie_schoolie)"],
        "trigger": ["sally \\(scalie schoolie\\), scalie schoolie"],
    },
    "lucas_(fuze)": {
        "character": ["lucas_(fuze)"],
        "trigger": ["lucas \\(fuze\\), pokemon"],
    },
    "brenna_jorunn": {
        "character": ["brenna_jorunn"],
        "trigger": ["brenna jorunn, mythology"],
    },
    "red_(shiro-neko)": {
        "character": ["red_(shiro-neko)"],
        "trigger": ["red \\(shiro-neko\\), pokemon"],
    },
    "dizzy_(dizzymilky)": {
        "character": ["dizzy_(dizzymilky)"],
        "trigger": ["dizzy \\(dizzymilky\\), pokemon"],
    },
    "draegonis": {"character": ["draegonis"], "trigger": ["draegonis, mythology"]},
    "orinette_(ceehaz)": {
        "character": ["orinette_(ceehaz)"],
        "trigger": ["orinette \\(ceehaz\\), dog knight rpg"],
    },
    "rat_god_(mad_rat_dead)": {
        "character": ["rat_god_(mad_rat_dead)"],
        "trigger": ["rat god \\(mad rat dead\\), nippon ichi software"],
    },
    "midnight_snowstorm": {
        "character": ["midnight_snowstorm"],
        "trigger": ["midnight snowstorm, my little pony"],
    },
    "girlfriend_(fnf)": {
        "character": ["girlfriend_(fnf)"],
        "trigger": ["girlfriend \\(fnf\\), friday night funkin'"],
    },
    "zephyr_(bateleurs)": {
        "character": ["zephyr_(bateleurs)"],
        "trigger": ["zephyr \\(bateleurs\\), mythology"],
    },
    "warfare_lopunny": {
        "character": ["warfare_lopunny"],
        "trigger": ["warfare lopunny, pokemon"],
    },
    "kera": {"character": ["kera"], "trigger": ["kera, mythology"]},
    "taichi_kamiya": {
        "character": ["taichi_kamiya"],
        "trigger": ["taichi kamiya, digimon"],
    },
    "tigra": {"character": ["tigra"], "trigger": ["tigra, marvel"]},
    "funky_kong": {
        "character": ["funky_kong"],
        "trigger": ["funky kong, donkey kong \\(series\\)"],
    },
    "kammypup": {"character": ["kammypup"], "trigger": ["kammypup, pokemon"]},
    "non_toxic_(oc)": {
        "character": ["non_toxic_(oc)"],
        "trigger": ["non toxic \\(oc\\), my little pony"],
    },
    "wuffle": {"character": ["wuffle"], "trigger": ["wuffle, wuffle \\(webcomic\\)"]},
    "esther_(rinkai)": {
        "character": ["esther_(rinkai)"],
        "trigger": ["esther \\(rinkai\\), wizards of the coast"],
    },
    "raier_(unrealplace)": {
        "character": ["raier_(unrealplace)"],
        "trigger": ["raier \\(unrealplace\\), mythology"],
    },
    "quibble_pants_(mlp)": {
        "character": ["quibble_pants_(mlp)"],
        "trigger": ["quibble pants \\(mlp\\), my little pony"],
    },
    "saul_ashle": {"character": ["saul_ashle"], "trigger": ["saul ashle, pokemon"]},
    "anonym0use": {"character": ["anonym0use"], "trigger": ["anonym0use, mythology"]},
    "malt_marzipan": {
        "character": ["malt_marzipan"],
        "trigger": ["malt marzipan, fuga: melodies of steel"],
    },
    "noire_vala": {"character": ["noire_vala"], "trigger": ["noire vala, mythology"]},
    "warfare_blaze": {
        "character": ["warfare_blaze"],
        "trigger": ["warfare blaze, sonic the hedgehog \\(series\\)"],
    },
    "clay_calloway_(sing)": {
        "character": ["clay_calloway_(sing)"],
        "trigger": ["clay calloway \\(sing\\), illumination entertainment"],
    },
    "zayats_(wjyw)": {
        "character": ["zayats_(wjyw)"],
        "trigger": ["zayats \\(wjyw\\), soyuzmultfilm"],
    },
    "biscuit_(biscuits)": {
        "character": ["biscuit_(biscuits)"],
        "trigger": ["biscuit \\(biscuits\\), mythology"],
    },
    "mimi_tachikawa": {
        "character": ["mimi_tachikawa"],
        "trigger": ["mimi tachikawa, digimon"],
    },
    "urta": {"character": ["urta"], "trigger": ["urta, corruption of champions"]},
    "breezie_the_hedgehog": {
        "character": ["breezie_the_hedgehog"],
        "trigger": ["breezie the hedgehog, sonic the hedgehog \\(series\\)"],
    },
    "cynder_nightshadow": {
        "character": ["cynder_nightshadow"],
        "trigger": ["cynder nightshadow, mythology"],
    },
    "scp-049": {"character": ["scp-049"], "trigger": ["scp-049, scp foundation"]},
    "kobold_princess": {
        "character": ["kobold_princess"],
        "trigger": ["kobold princess, towergirls"],
    },
    "fu_dog_(character)": {
        "character": ["fu_dog_(character)"],
        "trigger": ["fu dog \\(character\\), disney"],
    },
    "bastion_(overwatch)": {
        "character": ["bastion_(overwatch)"],
        "trigger": ["bastion \\(overwatch\\), overwatch"],
    },
    "huntress_wizard": {
        "character": ["huntress_wizard"],
        "trigger": ["huntress wizard, cartoon network"],
    },
    "ral": {"character": ["ral"], "trigger": ["ral, mythology"]},
    "carol_(hladilnik)": {
        "character": ["carol_(hladilnik)"],
        "trigger": ["carol \\(hladilnik\\), scp foundation"],
    },
    "freya_howell": {
        "character": ["freya_howell"],
        "trigger": ["freya howell, pokemon"],
    },
    "lemmy_(lemmy_niscuit)": {
        "character": ["lemmy_(lemmy_niscuit)"],
        "trigger": ["lemmy \\(lemmy niscuit\\), nintendo"],
    },
    "rita_(fuf)": {"character": ["rita_(fuf)"], "trigger": ["rita \\(fuf\\), pokemon"]},
    "bunzo_bunny": {
        "character": ["bunzo_bunny"],
        "trigger": ["bunzo bunny, poppy playtime"],
    },
    "nahida_(genshin_impact)": {
        "character": ["nahida_(genshin_impact)"],
        "trigger": ["nahida \\(genshin impact\\), mihoyo"],
    },
    "weighted_companion_cube": {
        "character": ["weighted_companion_cube"],
        "trigger": ["weighted companion cube, valve"],
    },
    "graedius_(character)": {
        "character": ["graedius_(character)"],
        "trigger": ["graedius \\(character\\), nintendo"],
    },
    "randal_hawthorne": {
        "character": ["randal_hawthorne"],
        "trigger": ["randal hawthorne, las lindas"],
    },
    "rangarig_rex": {
        "character": ["rangarig_rex"],
        "trigger": ["rangarig rex, mythology"],
    },
    "sorceress_(dragon's_crown)": {
        "character": ["sorceress_(dragon's_crown)"],
        "trigger": ["sorceress \\(dragon's crown\\), dragon's crown"],
    },
    "markiplier": {"character": ["markiplier"], "trigger": ["markiplier, scottgames"]},
    "redrick_(erickredfox)": {
        "character": ["redrick_(erickredfox)"],
        "trigger": ["redrick \\(erickredfox\\), digimon"],
    },
    "bibi_(o-den)": {
        "character": ["bibi_(o-den)"],
        "trigger": ["bibi \\(o-den\\), monster hunter"],
    },
    "nestor_(spyro)": {
        "character": ["nestor_(spyro)"],
        "trigger": ["nestor \\(spyro\\), mythology"],
    },
    "naz_namaki_(cynxie)": {
        "character": ["naz_namaki_(cynxie)"],
        "trigger": ["naz namaki \\(cynxie\\), christmas"],
    },
    "russel_(pickles-hyena)": {
        "character": ["russel_(pickles-hyena)"],
        "trigger": ["russel \\(pickles-hyena\\), family tensions"],
    },
    "fisk_cerris": {
        "character": ["fisk_cerris"],
        "trigger": ["fisk cerris, mythology"],
    },
    "jessica_young_melis": {
        "character": ["jessica_young_melis"],
        "trigger": ["jessica young melis, suinmsg"],
    },
    "shanukk": {"character": ["shanukk"], "trigger": ["shanukk, mythology"]},
    "entropy_(billeur)": {
        "character": ["entropy_(billeur)"],
        "trigger": ["entropy \\(billeur\\), mythology"],
    },
    "abby_(polyle)": {
        "character": ["abby_(polyle)"],
        "trigger": ["abby \\(polyle\\), piczel"],
    },
    "sun_wukong": {
        "character": ["sun_wukong"],
        "trigger": ["sun wukong, journey to the west"],
    },
    "throttle_(bmfm)": {
        "character": ["throttle_(bmfm)"],
        "trigger": ["throttle \\(bmfm\\), biker mice from mars"],
    },
    "pinky_(warner_brothers)": {
        "character": ["pinky_(warner_brothers)"],
        "trigger": ["pinky \\(warner brothers\\), warner brothers"],
    },
    "miss_bianca_(the_rescuers)": {
        "character": ["miss_bianca_(the_rescuers)"],
        "trigger": ["miss bianca \\(the rescuers\\), disney"],
    },
    "haku_(spirited_away)": {
        "character": ["haku_(spirited_away)"],
        "trigger": ["haku \\(spirited away\\), ghibli"],
    },
    "choko_(chokodonkey)": {
        "character": ["choko_(chokodonkey)"],
        "trigger": ["choko \\(chokodonkey\\), digimon"],
    },
    "wight_bracken": {
        "character": ["wight_bracken"],
        "trigger": ["wight bracken, monochrome \\(series\\)"],
    },
    "eileen_roberts": {
        "character": ["eileen_roberts"],
        "trigger": ["eileen roberts, cartoon network"],
    },
    "twilight_stormshi_(character)": {
        "character": ["twilight_stormshi_(character)"],
        "trigger": ["twilight stormshi \\(character\\), mario bros"],
    },
    "donk_(hladilnik)": {
        "character": ["donk_(hladilnik)"],
        "trigger": ["donk \\(hladilnik\\), my little pony"],
    },
    "boon_(vimhomeless)": {
        "character": ["boon_(vimhomeless)"],
        "trigger": ["boon \\(vimhomeless\\), mythology"],
    },
    "elmelie": {"character": ["elmelie"], "trigger": ["elmelie, mythology"]},
    "jessica_vega": {
        "character": ["jessica_vega"],
        "trigger": ["jessica vega, warcraft"],
    },
    "emery_waldren": {
        "character": ["emery_waldren"],
        "trigger": ["emery waldren, anthem"],
    },
    "germ_warfare_(nitw)": {
        "character": ["germ_warfare_(nitw)"],
        "trigger": ["germ warfare \\(nitw\\), night in the woods"],
    },
    "warning_(fluff-kevlar)": {
        "character": ["warning_(fluff-kevlar)"],
        "trigger": ["warning \\(fluff-kevlar\\), patreon"],
    },
    "aethrus": {"character": ["aethrus"], "trigger": ["aethrus, mythology"]},
    "darnell_(buddy_thunderstruck)": {
        "character": ["darnell_(buddy_thunderstruck)"],
        "trigger": [
            "darnell \\(buddy thunderstruck\\), buddy thunderstruck \\(series\\)"
        ],
    },
    "blue_(shiro-neko)": {
        "character": ["blue_(shiro-neko)"],
        "trigger": ["blue \\(shiro-neko\\), pokemon"],
    },
    "trainer_iris": {
        "character": ["trainer_iris"],
        "trigger": ["trainer iris, nintendo"],
    },
    "nyselia": {
        "character": ["nyselia"],
        "trigger": ["nyselia, monster girl encyclopedia"],
    },
    "o_(takahirosi)": {
        "character": ["o_(takahirosi)"],
        "trigger": ["o \\(takahirosi\\), patreon"],
    },
    "marzipan_(spottedtigress)": {
        "character": ["marzipan_(spottedtigress)"],
        "trigger": ["marzipan \\(spottedtigress\\), mythology"],
    },
    "firepawdacat": {
        "character": ["firepawdacat"],
        "trigger": ["firepawdacat, mythology"],
    },
    "rose_duskclaw": {
        "character": ["rose_duskclaw"],
        "trigger": ["rose duskclaw, pokemon"],
    },
    "tolng": {"character": ["tolng"], "trigger": ["tolng, mythology"]},
    "shina_(daigo)": {
        "character": ["shina_(daigo)"],
        "trigger": ["shina \\(daigo\\), patreon"],
    },
    "vikna_(fluff-kevlar)": {
        "character": ["vikna_(fluff-kevlar)"],
        "trigger": ["vikna \\(fluff-kevlar\\), christmas"],
    },
    "ada_wong": {"character": ["ada_wong"], "trigger": ["ada wong, resident evil"]},
    "falcon_mccooper_(character)": {
        "character": ["falcon_mccooper_(character)"],
        "trigger": ["falcon mccooper \\(character\\), mythology"],
    },
    "snoopy": {"character": ["snoopy"], "trigger": ["snoopy, peanuts \\(comic\\)"]},
    "reizo": {"character": ["reizo"], "trigger": ["reizo, tsukasa-spirit-fox"]},
    "mona_lisa_(tmnt)": {
        "character": ["mona_lisa_(tmnt)"],
        "trigger": ["mona lisa \\(tmnt\\), teenage mutant ninja turtles"],
    },
    "grinch": {
        "character": ["grinch"],
        "trigger": ["grinch, how the grinch stole christmas!"],
    },
    "jamie_the_oryx": {
        "character": ["jamie_the_oryx"],
        "trigger": ["jamie the oryx, mythology"],
    },
    "candy_(mrmadhead)": {
        "character": ["candy_(mrmadhead)"],
        "trigger": ["candy \\(mrmadhead\\), mythology"],
    },
    "alvo_(target_miss)": {
        "character": ["alvo_(target_miss)"],
        "trigger": ["alvo \\(target miss\\), target miss"],
    },
    "insomni": {"character": ["insomni"], "trigger": ["insomni, yo-kai watch"]},
    "trout_(character)": {
        "character": ["trout_(character)"],
        "trigger": ["trout \\(character\\), mythology"],
    },
    "damian_(zoophobia)": {
        "character": ["damian_(zoophobia)"],
        "trigger": ["damian \\(zoophobia\\), zoophobia"],
    },
    "pietro_(felino)": {
        "character": ["pietro_(felino)"],
        "trigger": ["pietro \\(felino\\), nintendo"],
    },
    "fylk": {"character": ["fylk"], "trigger": ["fylk, the onion"]},
    "edwin_inculous_(character)": {
        "character": ["edwin_inculous_(character)"],
        "trigger": ["edwin inculous \\(character\\), nintendo"],
    },
    "tapio_chatarozawa": {
        "character": ["tapio_chatarozawa"],
        "trigger": ["tapio chatarozawa, working buddies!"],
    },
    "dart_(thecon)": {
        "character": ["dart_(thecon)"],
        "trigger": ["dart \\(thecon\\), christmas"],
    },
    "stan_(beez)": {
        "character": ["stan_(beez)"],
        "trigger": ["stan \\(beez\\), patreon"],
    },
    "elfilin": {"character": ["elfilin"], "trigger": ["elfilin, kirby \\(series\\)"]},
    "bart_simpson": {
        "character": ["bart_simpson"],
        "trigger": ["bart simpson, the simpsons"],
    },
    "leatherhead": {
        "character": ["leatherhead"],
        "trigger": ["leatherhead, teenage mutant ninja turtles"],
    },
    "fliqpy_(htf)": {
        "character": ["fliqpy_(htf)"],
        "trigger": ["fliqpy \\(htf\\), happy tree friends"],
    },
    "tootsie": {"character": ["tootsie"], "trigger": ["tootsie, las lindas"]},
    "dotty_(animal_crossing)": {
        "character": ["dotty_(animal_crossing)"],
        "trigger": ["dotty \\(animal crossing\\), animal crossing"],
    },
    "drum's_father": {
        "character": ["drum's_father"],
        "trigger": ["drum's father, future card buddyfight"],
    },
    "scrump": {"character": ["scrump"], "trigger": ["scrump, disney"]},
    "generic_messy_hair_anime_anon": {
        "character": ["generic_messy_hair_anime_anon"],
        "trigger": ["generic messy hair anime anon, my little pony"],
    },
    "dusk_(tabuley)": {
        "character": ["dusk_(tabuley)"],
        "trigger": ["dusk \\(tabuley\\), mythology"],
    },
    "arcade_bunny": {
        "character": ["arcade_bunny"],
        "trigger": ["arcade bunny, nintendo badge arcade"],
    },
    "cathyl_(monster_musume)": {
        "character": ["cathyl_(monster_musume)"],
        "trigger": ["cathyl \\(monster musume\\), monster musume"],
    },
    "shouta_magatsuchi": {
        "character": ["shouta_magatsuchi"],
        "trigger": ["shouta magatsuchi, miss kobayashi's dragon maid"],
    },
    "hollow_knight_(character)": {
        "character": ["hollow_knight_(character)"],
        "trigger": ["hollow knight \\(character\\), team cherry"],
    },
    "heat_(lilo_and_stitch)": {
        "character": ["heat_(lilo_and_stitch)"],
        "trigger": ["heat \\(lilo and stitch\\), disney"],
    },
    "pseftis_savra": {
        "character": ["pseftis_savra"],
        "trigger": ["pseftis savra, mythology"],
    },
    "eugeniy_g": {"character": ["eugeniy_g"], "trigger": ["eugeniy g, twokinds"]},
    "niko_(pkfirefawx)": {
        "character": ["niko_(pkfirefawx)"],
        "trigger": ["niko \\(pkfirefawx\\), nintendo"],
    },
    "octavian_(7intandadream)": {
        "character": ["octavian_(7intandadream)"],
        "trigger": ["octavian \\(7intandadream\\), mythology"],
    },
    "archon_eclipse": {
        "character": ["archon_eclipse"],
        "trigger": ["archon eclipse, pokemon"],
    },
    "sabbyth": {"character": ["sabbyth"], "trigger": ["sabbyth, mythology"]},
    "elinor_rabbit": {
        "character": ["elinor_rabbit"],
        "trigger": ["elinor rabbit, elinor wonders why"],
    },
    "yaita_(character)": {
        "character": ["yaita_(character)"],
        "trigger": ["yaita \\(character\\), mythology"],
    },
    "issun_(okami)": {
        "character": ["issun_(okami)"],
        "trigger": ["issun \\(okami\\), okami \\(capcom\\)"],
    },
    "xander_the_blue": {
        "character": ["xander_the_blue"],
        "trigger": ["xander the blue, mythology"],
    },
    "bowser_koopa_junior_(roommates)": {
        "character": ["bowser_koopa_junior_(roommates)"],
        "trigger": ["bowser koopa junior \\(roommates\\), mario bros"],
    },
    "flandre_scarlet": {
        "character": ["flandre_scarlet"],
        "trigger": ["flandre scarlet, touhou"],
    },
    "perry_the_platypus": {
        "character": ["perry_the_platypus"],
        "trigger": ["perry the platypus, disney"],
    },
    "macharius": {"character": ["macharius"], "trigger": ["macharius, mythology"]},
    "bella_(animal_crossing)": {
        "character": ["bella_(animal_crossing)"],
        "trigger": ["bella \\(animal crossing\\), animal crossing"],
    },
    "zuma_(paw_patrol)": {
        "character": ["zuma_(paw_patrol)"],
        "trigger": ["zuma \\(paw patrol\\), paw patrol"],
    },
    "flash_slothmore": {
        "character": ["flash_slothmore"],
        "trigger": ["flash slothmore, disney"],
    },
    "geno_sans_(aftertale)": {
        "character": ["geno_sans_(aftertale)"],
        "trigger": ["geno sans \\(aftertale\\), aftertale"],
    },
    "larry_(zootopia)": {
        "character": ["larry_(zootopia)"],
        "trigger": ["larry \\(zootopia\\), disney"],
    },
    "saffron_masala_(mlp)": {
        "character": ["saffron_masala_(mlp)"],
        "trigger": ["saffron masala \\(mlp\\), my little pony"],
    },
    "riley_(scratch21)": {
        "character": ["riley_(scratch21)"],
        "trigger": ["riley \\(scratch21\\), scratch21"],
    },
    "rowrow": {"character": ["rowrow"], "trigger": ["rowrow, mythology"]},
    "crushfang_(sdorica_sunset)": {
        "character": ["crushfang_(sdorica_sunset)"],
        "trigger": ["crushfang \\(sdorica sunset\\), sdorica"],
    },
    "durham_(beastars)": {
        "character": ["durham_(beastars)"],
        "trigger": ["durham \\(beastars\\), beastars"],
    },
    "livia_(dreamypride)": {
        "character": ["livia_(dreamypride)"],
        "trigger": ["livia \\(dreamypride\\), twitter"],
    },
    "kanna_(joaoppereiraus)": {
        "character": ["kanna_(joaoppereiraus)"],
        "trigger": ["kanna \\(joaoppereiraus\\), pokemon"],
    },
    "jeffrey_taggart": {
        "character": ["jeffrey_taggart"],
        "trigger": ["jeffrey taggart, no nut november"],
    },
    "lena_fluffy_(character)": {
        "character": ["lena_fluffy_(character)"],
        "trigger": ["lena fluffy \\(character\\), atomic heart"],
    },
    "tak_(invader_zim)": {
        "character": ["tak_(invader_zim)"],
        "trigger": ["tak \\(invader zim\\), invader zim"],
    },
    "road_runner_(looney_tunes)": {
        "character": ["road_runner_(looney_tunes)"],
        "trigger": ["road runner \\(looney tunes\\), warner brothers"],
    },
    "felix_the_cat": {
        "character": ["felix_the_cat"],
        "trigger": ["felix the cat, felix the cat \\(series\\)"],
    },
    "stacey_skunkette": {
        "character": ["stacey_skunkette"],
        "trigger": ["stacey skunkette, furafterdark"],
    },
    "sora_takenouchi": {
        "character": ["sora_takenouchi"],
        "trigger": ["sora takenouchi, digimon"],
    },
    "nico_robin": {"character": ["nico_robin"], "trigger": ["nico robin, one piece"]},
    "ripper_roo": {
        "character": ["ripper_roo"],
        "trigger": ["ripper roo, crash bandicoot \\(series\\)"],
    },
    "sesame_akane": {
        "character": ["sesame_akane"],
        "trigger": ["sesame akane, uberquest"],
    },
    "fols": {"character": ["fols"], "trigger": ["fols, mythology"]},
    "mike_(bcb)": {
        "character": ["mike_(bcb)"],
        "trigger": ["mike \\(bcb\\), bittersweet candy bowl"],
    },
    "ignitus": {"character": ["ignitus"], "trigger": ["ignitus, spyro the dragon"]},
    "shyloc": {"character": ["shyloc"], "trigger": ["shyloc, mythology"]},
    "tharis": {"character": ["tharis"], "trigger": ["tharis, mythology"]},
    "trafalgar_law": {
        "character": ["trafalgar_law"],
        "trigger": ["trafalgar law, one piece"],
    },
    "gooze_(sunibee)": {
        "character": ["gooze_(sunibee)"],
        "trigger": ["gooze \\(sunibee\\), pokemon"],
    },
    "suri_polomare_(mlp)": {
        "character": ["suri_polomare_(mlp)"],
        "trigger": ["suri polomare \\(mlp\\), my little pony"],
    },
    "misskari": {"character": ["misskari"], "trigger": ["misskari, patreon"]},
    "thaz_(character)": {
        "character": ["thaz_(character)"],
        "trigger": ["thaz \\(character\\), mythology"],
    },
    "zuthal": {"character": ["zuthal"], "trigger": ["zuthal, my little pony"]},
    "hakuro_(onmyoji)": {
        "character": ["hakuro_(onmyoji)"],
        "trigger": ["hakuro \\(onmyoji\\), onmyoji"],
    },
    "kraft_trio": {"character": ["kraft_trio"], "trigger": ["kraft trio, bad dragon"]},
    "ruby_(ultilix)": {
        "character": ["ruby_(ultilix)"],
        "trigger": ["ruby \\(ultilix\\), mythology"],
    },
    "vinci_(itsmemtfo4)": {
        "character": ["vinci_(itsmemtfo4)"],
        "trigger": ["vinci \\(itsmemtfo4\\), pokemon"],
    },
    "abby_(canisfidelis)": {
        "character": ["abby_(canisfidelis)"],
        "trigger": ["abby \\(canisfidelis\\), no nut november"],
    },
    "tini_(grimart)": {
        "character": ["tini_(grimart)"],
        "trigger": ["tini \\(grimart\\), pokemon"],
    },
    "ammit_(moon_knight)": {
        "character": ["ammit_(moon_knight)"],
        "trigger": ["ammit \\(moon knight\\), mythology"],
    },
    "sherry_bear_(blarf022)": {
        "character": ["sherry_bear_(blarf022)"],
        "trigger": ["sherry bear \\(blarf022\\), christmas"],
    },
    "alma_(vixinecomics)": {
        "character": ["alma_(vixinecomics)"],
        "trigger": ["alma \\(vixinecomics\\), quest for fun"],
    },
    "beast_(marvel)": {
        "character": ["beast_(marvel)"],
        "trigger": ["beast \\(marvel\\), marvel"],
    },
    "rover_(mlp)": {
        "character": ["rover_(mlp)"],
        "trigger": ["rover \\(mlp\\), my little pony"],
    },
    "thistle_(frisky_ferals)": {
        "character": ["thistle_(frisky_ferals)"],
        "trigger": ["thistle \\(frisky ferals\\), mythology"],
    },
    "skoon_(character)": {
        "character": ["skoon_(character)"],
        "trigger": ["skoon \\(character\\), mythology"],
    },
    "apophis_(mge)": {
        "character": ["apophis_(mge)"],
        "trigger": ["apophis \\(mge\\), monster girl encyclopedia"],
    },
    "muto_(godzilla)": {
        "character": ["muto_(godzilla)"],
        "trigger": ["muto \\(godzilla\\), godzilla \\(series\\)"],
    },
    "dyson_(eldiman)": {
        "character": ["dyson_(eldiman)"],
        "trigger": ["dyson \\(eldiman\\), halloween"],
    },
    "zoey_(brushfire)": {
        "character": ["zoey_(brushfire)"],
        "trigger": ["zoey \\(brushfire\\), mythology"],
    },
    "pronk_oryx-antlerson": {
        "character": ["pronk_oryx-antlerson"],
        "trigger": ["pronk oryx-antlerson, disney"],
    },
    "perro-kun": {"character": ["perro-kun"], "trigger": ["perro-kun, live2d"]},
    "alric_kyznetsov": {
        "character": ["alric_kyznetsov"],
        "trigger": ["alric kyznetsov, halloween"],
    },
    "shoebill_(kemono_friends)": {
        "character": ["shoebill_(kemono_friends)"],
        "trigger": ["shoebill \\(kemono friends\\), kemono friends"],
    },
    "resasuke": {"character": ["resasuke"], "trigger": ["resasuke, sanrio"]},
    "nano_(nanosheep)": {
        "character": ["nano_(nanosheep)"],
        "trigger": ["nano \\(nanosheep\\), pokemon"],
    },
    "jewel_the_beetle": {
        "character": ["jewel_the_beetle"],
        "trigger": ["jewel the beetle, sonic the hedgehog \\(series\\)"],
    },
    "solii_(gizmo1205)": {
        "character": ["solii_(gizmo1205)"],
        "trigger": ["solii \\(gizmo1205\\), fallout"],
    },
    "sebastian_cummins_(thechavicgerman)": {
        "character": ["sebastian_cummins_(thechavicgerman)"],
        "trigger": ["sebastian cummins \\(thechavicgerman\\), halloween"],
    },
    "dauna_(reptilligator)": {
        "character": ["dauna_(reptilligator)"],
        "trigger": ["dauna \\(reptilligator\\), capcom"],
    },
    "jace_(darknaig)": {
        "character": ["jace_(darknaig)"],
        "trigger": ["jace \\(darknaig\\), universal studios"],
    },
    "pyramid_head_(silent_hill)": {
        "character": ["pyramid_head_(silent_hill)"],
        "trigger": ["pyramid head \\(silent hill\\), silent hill"],
    },
    "garr_(breath_of_fire)": {
        "character": ["garr_(breath_of_fire)"],
        "trigger": ["garr \\(breath of fire\\), breath of fire"],
    },
    "colin_young": {
        "character": ["colin_young"],
        "trigger": ["colin young, closet coon"],
    },
    "kitty_(hayakain)": {
        "character": ["kitty_(hayakain)"],
        "trigger": ["kitty \\(hayakain\\), sonic the hedgehog \\(series\\)"],
    },
    "smaug": {"character": ["smaug"], "trigger": ["smaug, middle-earth \\(tolkien\\)"]},
    "filthy_rich_(mlp)": {
        "character": ["filthy_rich_(mlp)"],
        "trigger": ["filthy rich \\(mlp\\), my little pony"],
    },
    "tina_russo": {
        "character": ["tina_russo"],
        "trigger": ["tina russo, warner brothers"],
    },
    "sarki_(character)": {
        "character": ["sarki_(character)"],
        "trigger": ["sarki \\(character\\), mythology"],
    },
    "senketsu": {"character": ["senketsu"], "trigger": ["senketsu, kill la kill"]},
    "ruze": {"character": ["ruze"], "trigger": ["ruze, mythology"]},
    "moonbrush_(phathusa)": {
        "character": ["moonbrush_(phathusa)"],
        "trigger": ["moonbrush \\(phathusa\\), my little pony"],
    },
    "ladon_(character)": {
        "character": ["ladon_(character)"],
        "trigger": ["ladon \\(character\\), mythology"],
    },
    "frori": {"character": ["frori"], "trigger": ["frori, mythology"]},
    "sascha_(hypnofood)": {
        "character": ["sascha_(hypnofood)"],
        "trigger": ["sascha \\(hypnofood\\), the jungle book"],
    },
    "sten_fletcher": {
        "character": ["sten_fletcher"],
        "trigger": ["sten fletcher, nintendo"],
    },
    "aster_faye": {"character": ["aster_faye"], "trigger": ["aster faye, nintendo"]},
    "nova_(anonym0use)": {
        "character": ["nova_(anonym0use)"],
        "trigger": ["nova \\(anonym0use\\), mythology"],
    },
    "hibiscus_blossom": {
        "character": ["hibiscus_blossom"],
        "trigger": ["hibiscus blossom, hasbro"],
    },
    "okami_bark": {"character": ["okami_bark"], "trigger": ["okami bark, christmas"]},
    "ed_(scratch21)": {
        "character": ["ed_(scratch21)"],
        "trigger": ["ed \\(scratch21\\), scratch21"],
    },
    "rezukii": {"character": ["rezukii"], "trigger": ["rezukii, mythology"]},
    "diona_(genshin_impact)": {
        "character": ["diona_(genshin_impact)"],
        "trigger": ["diona \\(genshin impact\\), mihoyo"],
    },
    "power_(chainsaw_man)": {
        "character": ["power_(chainsaw_man)"],
        "trigger": ["power \\(chainsaw man\\), chainsaw man"],
    },
    "midnite_(mario_plus_rabbids)": {
        "character": ["midnite_(mario_plus_rabbids)"],
        "trigger": ["midnite \\(mario plus rabbids\\), raving rabbids"],
    },
    "zidane_tribal": {
        "character": ["zidane_tribal"],
        "trigger": ["zidane tribal, square enix"],
    },
    "broody": {"character": ["broody"], "trigger": ["broody, theta iota kappa"]},
    "andy_porter": {
        "character": ["andy_porter"],
        "trigger": ["andy porter, webcame \\(comic\\)"],
    },
    "remilia_scarlet": {
        "character": ["remilia_scarlet"],
        "trigger": ["remilia scarlet, touhou"],
    },
    "nurse_joy": {"character": ["nurse_joy"], "trigger": ["nurse joy, pokemon"]},
    "greywolf_blacksock": {
        "character": ["greywolf_blacksock"],
        "trigger": ["greywolf blacksock, mythology"],
    },
    "bill_cipher": {"character": ["bill_cipher"], "trigger": ["bill cipher, disney"]},
    "agent_classified": {
        "character": ["agent_classified"],
        "trigger": ["agent classified, the penguins of madagascar"],
    },
    "purple_man_(fnaf)": {
        "character": ["purple_man_(fnaf)"],
        "trigger": ["purple man \\(fnaf\\), scottgames"],
    },
    "wojak": {"character": ["wojak"], "trigger": ["wojak, coomer wojak"]},
    "bahati_whiteclaw": {
        "character": ["bahati_whiteclaw"],
        "trigger": ["bahati whiteclaw, disney"],
    },
    "alex_(beez)": {
        "character": ["alex_(beez)"],
        "trigger": ["alex \\(beez\\), patreon"],
    },
    "davad_(odissy)": {
        "character": ["davad_(odissy)"],
        "trigger": ["davad \\(odissy\\), mythology"],
    },
    "alex_(phrannd)": {
        "character": ["alex_(phrannd)"],
        "trigger": ["alex \\(phrannd\\), mythology"],
    },
    "bow_(bowhuskers)": {
        "character": ["bow_(bowhuskers)"],
        "trigger": ["bow \\(bowhuskers\\), pokemon"],
    },
    "leo_(domibun)": {
        "character": ["leo_(domibun)"],
        "trigger": ["leo \\(domibun\\), source filmmaker"],
    },
    "ragatha_(tadc)": {
        "character": ["ragatha_(tadc)"],
        "trigger": ["ragatha \\(tadc\\), the amazing digital circus"],
    },
    "quote_(cave_story)": {
        "character": ["quote_(cave_story)"],
        "trigger": ["quote \\(cave story\\), cave story"],
    },
    "suu_(monster_musume)": {
        "character": ["suu_(monster_musume)"],
        "trigger": ["suu \\(monster musume\\), monster musume"],
    },
    "xenia_(linux)": {
        "character": ["xenia_(linux)"],
        "trigger": ["xenia \\(linux\\), linux"],
    },
    "elesa_(pokemon)": {
        "character": ["elesa_(pokemon)"],
        "trigger": ["elesa \\(pokemon\\), pokemon"],
    },
    "gwen_mason": {"character": ["gwen_mason"], "trigger": ["gwen mason, nintendo"]},
    "song_(kung_fu_panda)": {
        "character": ["song_(kung_fu_panda)"],
        "trigger": ["song \\(kung fu panda\\), kung fu panda"],
    },
    "tarke": {"character": ["tarke"], "trigger": ["tarke, werelion2003"]},
    "morgan_laranda": {
        "character": ["morgan_laranda"],
        "trigger": ["morgan laranda, tale of tails"],
    },
    "maxximizer": {"character": ["maxximizer"], "trigger": ["maxximizer, digimon"]},
    "giraffe_mom": {
        "character": ["giraffe_mom"],
        "trigger": ['giraffe mom, toys "r" us'],
    },
    "rick_sanchez": {
        "character": ["rick_sanchez"],
        "trigger": ["rick sanchez, rick and morty"],
    },
    "nicoya": {"character": ["nicoya"], "trigger": ["nicoya, nintendo"]},
    "james_auvereign": {
        "character": ["james_auvereign"],
        "trigger": ["james auvereign, mythology"],
    },
    "durabelle": {"character": ["durabelle"], "trigger": ["durabelle, mythology"]},
    "abbi_(kilinah)": {
        "character": ["abbi_(kilinah)"],
        "trigger": ["abbi \\(kilinah\\), dust to dust"],
    },
    "maggie_applebee": {
        "character": ["maggie_applebee"],
        "trigger": ["maggie applebee, christmas"],
    },
    "kazu_(thatgaysushi)": {
        "character": ["kazu_(thatgaysushi)"],
        "trigger": ["kazu \\(thatgaysushi\\), mythology"],
    },
    "king_spade": {
        "character": ["king_spade"],
        "trigger": ["king spade, undertale \\(series\\)"],
    },
    "sam_(pink)": {
        "character": ["sam_(pink)"],
        "trigger": ["sam \\(pink\\), da silva"],
    },
    "barbie_(helluva_boss)": {
        "character": ["barbie_(helluva_boss)"],
        "trigger": ["barbie \\(helluva boss\\), helluva boss"],
    },
    "perec": {"character": ["perec"], "trigger": ["perec, mythology"]},
    "kai_(mr.smile)": {
        "character": ["kai_(mr.smile)"],
        "trigger": ["kai \\(mr.smile\\), pokemon"],
    },
    "feirune": {
        "character": ["feirune"],
        "trigger": ["feirune, so i'm a spider so what?"],
    },
    "emily_(paddedulf)": {
        "character": ["emily_(paddedulf)"],
        "trigger": ["emily \\(paddedulf\\), christmas"],
    },
    "lucy_(thisaccountdoesexist)": {
        "character": ["lucy_(thisaccountdoesexist)"],
        "trigger": ["lucy \\(thisaccountdoesexist\\), halloween"],
    },
    "isaacjexo": {
        "character": ["isaacjexo"],
        "trigger": ["isaacjexo, blender \\(software\\)"],
    },
    "bray_(kitfox-crimson)": {
        "character": ["bray_(kitfox-crimson)"],
        "trigger": ["bray \\(kitfox-crimson\\), in our shadow"],
    },
    "sarafina_(the_lion_king)": {
        "character": ["sarafina_(the_lion_king)"],
        "trigger": ["sarafina \\(the lion king\\), disney"],
    },
    "pluto_(disney)": {
        "character": ["pluto_(disney)"],
        "trigger": ["pluto \\(disney\\), disney"],
    },
    "tila_sunrise": {
        "character": ["tila_sunrise"],
        "trigger": ["tila sunrise, las lindas"],
    },
    "wolverine_(marvel)": {
        "character": ["wolverine_(marvel)"],
        "trigger": ["wolverine \\(marvel\\), marvel"],
    },
    "tiffany_(animal_crossing)": {
        "character": ["tiffany_(animal_crossing)"],
        "trigger": ["tiffany \\(animal crossing\\), animal crossing"],
    },
    "vex_(donryu)": {
        "character": ["vex_(donryu)"],
        "trigger": ["vex \\(donryu\\), nintendo"],
    },
    "master_monkey": {
        "character": ["master_monkey"],
        "trigger": ["master monkey, kung fu panda"],
    },
    "acino": {"character": ["acino"], "trigger": ["acino, team rocket"]},
    "awkore": {"character": ["awkore"], "trigger": ["awkore, nintendo"]},
    "vishka": {"character": ["vishka"], "trigger": ["vishka, patreon"]},
    "spooky_(sjm)": {
        "character": ["spooky_(sjm)"],
        "trigger": ["spooky \\(sjm\\), spooky's jump scare mansion"],
    },
    "twixxel_minty": {
        "character": ["twixxel_minty"],
        "trigger": ["twixxel minty, mythology"],
    },
    "adrian_iliovici": {
        "character": ["adrian_iliovici"],
        "trigger": ["adrian iliovici, patreon"],
    },
    "bucephalus": {"character": ["bucephalus"], "trigger": ["bucephalus, mythology"]},
    "atago_(azur_lane)": {
        "character": ["atago_(azur_lane)"],
        "trigger": ["atago \\(azur lane\\), azur lane"],
    },
    "elizabeth_fox": {
        "character": ["elizabeth_fox"],
        "trigger": ["elizabeth fox, i mean breast milk"],
    },
    "yao_(sdorica)": {
        "character": ["yao_(sdorica)"],
        "trigger": ["yao \\(sdorica\\), sdorica"],
    },
    "tabitha_(sabrina_online)": {
        "character": ["tabitha_(sabrina_online)"],
        "trigger": ["tabitha \\(sabrina online\\), sabrina online"],
    },
    "nightcrawler": {"character": ["nightcrawler"], "trigger": ["nightcrawler, x-men"]},
    "aiden_harris": {
        "character": ["aiden_harris"],
        "trigger": ["aiden harris, closet coon"],
    },
    "zone-tan": {"character": ["zone-tan"], "trigger": ["zone-tan, nintendo"]},
    "lux_(lol)": {"character": ["lux_(lol)"], "trigger": ["lux \\(lol\\), riot games"]},
    "ten_kodori": {"character": ["ten_kodori"], "trigger": ["ten kodori, morenatsu"]},
    "chrissy_(animal_crossing)": {
        "character": ["chrissy_(animal_crossing)"],
        "trigger": ["chrissy \\(animal crossing\\), animal crossing"],
    },
    "leo_(whiteleo)": {
        "character": ["leo_(whiteleo)"],
        "trigger": ["leo \\(whiteleo\\), halloween"],
    },
    "nate_(8chan)": {
        "character": ["nate_(8chan)"],
        "trigger": ["nate \\(8chan\\), 8chan"],
    },
    "thash": {"character": ["thash"], "trigger": ["thash, mythology"]},
    "zoru": {"character": ["zoru"], "trigger": ["zoru, warcraft"]},
    "river_(armello)": {
        "character": ["river_(armello)"],
        "trigger": ["river \\(armello\\), armello"],
    },
    "witch_doctor_(terraria)": {
        "character": ["witch_doctor_(terraria)"],
        "trigger": ["witch doctor \\(terraria\\), terraria"],
    },
    "seiya_(saku1saya)": {
        "character": ["seiya_(saku1saya)"],
        "trigger": ["seiya \\(saku1saya\\), scottgames"],
    },
    "mercury_shine": {
        "character": ["mercury_shine"],
        "trigger": ["mercury shine, my little pony"],
    },
    "monroe_lehner": {
        "character": ["monroe_lehner"],
        "trigger": ["monroe lehner, mythology"],
    },
    "skipsy_dragon_(character)": {
        "character": ["skipsy_dragon_(character)"],
        "trigger": ["skipsy dragon \\(character\\), mythology"],
    },
    "purna_whitewillow": {
        "character": ["purna_whitewillow"],
        "trigger": ["purna whitewillow, guild wars"],
    },
    "ash_(ashkelling)": {
        "character": ["ash_(ashkelling)"],
        "trigger": ["ash \\(ashkelling\\), twokinds"],
    },
    "geordie_79": {"character": ["geordie_79"], "trigger": ["geordie 79, mythology"]},
    "soleil_(itstedda)": {
        "character": ["soleil_(itstedda)"],
        "trigger": ["soleil \\(itstedda\\), mythology"],
    },
    "abe_(mikrogoat)": {
        "character": ["abe_(mikrogoat)"],
        "trigger": ["abe \\(mikrogoat\\), karen \\(meme\\)"],
    },
    "cali_(nastycalamari)": {
        "character": ["cali_(nastycalamari)"],
        "trigger": ["cali \\(nastycalamari\\), no nut november"],
    },
    "kerolink": {"character": ["kerolink"], "trigger": ["kerolink, nintendo"]},
    "ace_(claweddrip)": {
        "character": ["ace_(claweddrip)"],
        "trigger": ["ace \\(claweddrip\\), mythology"],
    },
    "lucas_(earthbound)": {
        "character": ["lucas_(earthbound)"],
        "trigger": ["lucas \\(earthbound\\), earthbound \\(series\\)"],
    },
    "blitz_(road_rovers)": {
        "character": ["blitz_(road_rovers)"],
        "trigger": ["blitz \\(road rovers\\), road rovers"],
    },
    "chowder": {"character": ["chowder"], "trigger": ["chowder, cartoon network"]},
    "miss_piggy": {"character": ["miss_piggy"], "trigger": ["miss piggy, muppets"]},
    "raider_(fallout)": {
        "character": ["raider_(fallout)"],
        "trigger": ["raider \\(fallout\\), bethesda softworks"],
    },
    "ahastar": {"character": ["ahastar"], "trigger": ["ahastar, digimon"]},
    "barzillai": {"character": ["barzillai"], "trigger": ["barzillai, mythology"]},
    "zhang_fei_(full_bokko_heroes)": {
        "character": ["zhang_fei_(full_bokko_heroes)"],
        "trigger": ["zhang fei \\(full bokko heroes\\), drecom"],
    },
    "lactaid_cow": {"character": ["lactaid_cow"], "trigger": ["lactaid cow, lactaid"]},
    "buru_(jaynatorburudragon)": {
        "character": ["buru_(jaynatorburudragon)"],
        "trigger": ["buru \\(jaynatorburudragon\\), mythology"],
    },
    "qwertyigloo": {
        "character": ["qwertyigloo"],
        "trigger": ["qwertyigloo, mythology"],
    },
    "wolter_(weaver)": {
        "character": ["wolter_(weaver)"],
        "trigger": ["wolter \\(weaver\\), pack street"],
    },
    "songbird_serenade_(mlp)": {
        "character": ["songbird_serenade_(mlp)"],
        "trigger": ["songbird serenade \\(mlp\\), my little pony"],
    },
    "kimmy_(felino)": {
        "character": ["kimmy_(felino)"],
        "trigger": ["kimmy \\(felino\\), nintendo"],
    },
    "carla_(ok_k.o.!_lbh)": {
        "character": ["carla_(ok_k.o.!_lbh)"],
        "trigger": ["carla \\(ok k.o.! lbh\\), cartoon network"],
    },
    "hazuki_mikami_(hasukii)": {
        "character": ["hazuki_mikami_(hasukii)"],
        "trigger": ["hazuki mikami \\(hasukii\\), valentine's day"],
    },
    "fhyra": {"character": ["fhyra"], "trigger": ["fhyra, fhyrrain"]},
    "elvia": {"character": ["elvia"], "trigger": ["elvia, wizards of the coast"]},
    "red_(among_us)": {
        "character": ["red_(among_us)"],
        "trigger": ["red \\(among us\\), among us"],
    },
    "llydian_(fyixen)": {
        "character": ["llydian_(fyixen)"],
        "trigger": ["llydian \\(fyixen\\), jeep"],
    },
    "the_one_who_waits": {
        "character": ["the_one_who_waits"],
        "trigger": ["the one who waits, cult of the lamb"],
    },
    "serket_(psychoh13)": {
        "character": ["serket_(psychoh13)"],
        "trigger": ["serket \\(psychoh13\\), mythology"],
    },
    "lexi_joyhart": {
        "character": ["lexi_joyhart"],
        "trigger": ["lexi joyhart, warcraft"],
    },
    "digidestined": {
        "character": ["digidestined"],
        "trigger": ["digidestined, digimon"],
    },
    "tewi_inaba": {"character": ["tewi_inaba"], "trigger": ["tewi inaba, touhou"]},
    "amione": {"character": ["amione"], "trigger": ["amione, mythology"]},
    "toroko": {"character": ["toroko"], "trigger": ["toroko, cave story"]},
    "opala_(legend_of_queen_opala)": {
        "character": ["opala_(legend_of_queen_opala)"],
        "trigger": ["opala \\(legend of queen opala\\), legend of queen opala"],
    },
    "angel_loveridge": {
        "character": ["angel_loveridge"],
        "trigger": ["angel loveridge, las lindas"],
    },
    "shari": {"character": ["shari"], "trigger": ["shari, mythology"]},
    "rolf": {"character": ["rolf"], "trigger": ["rolf, pokemon"]},
    "trixie_(jay_naylor)": {
        "character": ["trixie_(jay_naylor)"],
        "trigger": ["trixie \\(jay naylor\\), subscribestar"],
    },
    "thrakos": {"character": ["thrakos"], "trigger": ["thrakos, square enix"]},
    "flim_(mlp)": {
        "character": ["flim_(mlp)"],
        "trigger": ["flim \\(mlp\\), my little pony"],
    },
    "werethrope_laporte": {
        "character": ["werethrope_laporte"],
        "trigger": ["werethrope laporte, mythology"],
    },
    "mocha_latte": {
        "character": ["mocha_latte"],
        "trigger": ["mocha latte, my little pony"],
    },
    "twitter_bird": {
        "character": ["twitter_bird"],
        "trigger": ["twitter bird, twitter"],
    },
    "otake": {"character": ["otake"], "trigger": ["otake, christmas"]},
    "zak_(fvt)": {
        "character": ["zak_(fvt)"],
        "trigger": ["zak \\(fvt\\), fairies vs tentacles"],
    },
    "felicity_longis": {
        "character": ["felicity_longis"],
        "trigger": ["felicity longis, mythology"],
    },
    "vasira": {"character": ["vasira"], "trigger": ["vasira, my little pony"]},
    "kor'desse": {"character": ["kor'desse"], "trigger": ["kor'desse, mythology"]},
    "sir_pentious_(hazbin_hotel)": {
        "character": ["sir_pentious_(hazbin_hotel)"],
        "trigger": ["sir pentious \\(hazbin hotel\\), hazbin hotel"],
    },
    "sophring_jie": {
        "character": ["sophring_jie"],
        "trigger": ["sophring jie, full attack"],
    },
    "shanty_(tfh)": {
        "character": ["shanty_(tfh)"],
        "trigger": ["shanty \\(tfh\\), them's fightin' herds"],
    },
    "snuffy": {"character": ["snuffy"], "trigger": ["snuffy, vtuber"]},
    "tracey_tailor": {
        "character": ["tracey_tailor"],
        "trigger": ["tracey tailor, furafterdark"],
    },
    "philomena_(mlp)": {
        "character": ["philomena_(mlp)"],
        "trigger": ["philomena \\(mlp\\), my little pony"],
    },
    "pokemon_breeder": {
        "character": ["pokemon_breeder"],
        "trigger": ["pokemon breeder, pokemon"],
    },
    "pegaslut": {"character": ["pegaslut"], "trigger": ["pegaslut, my little pony"]},
    "meryl": {"character": ["meryl"], "trigger": ["meryl, pokemon"]},
    "ziggy_(dezo)": {
        "character": ["ziggy_(dezo)"],
        "trigger": ["ziggy \\(dezo\\), dezo"],
    },
    "tracy_(sailoranna)": {
        "character": ["tracy_(sailoranna)"],
        "trigger": ["tracy \\(sailoranna\\), halloween"],
    },
    "jv": {"character": ["jv"], "trigger": ["jv, mythology"]},
    "anneke_(weaver)": {
        "character": ["anneke_(weaver)"],
        "trigger": ["anneke \\(weaver\\), pack street"],
    },
    "zazush_(zazush-una)": {
        "character": ["zazush_(zazush-una)"],
        "trigger": ["zazush \\(zazush-una\\), mythology"],
    },
    "blaze_wolf": {
        "character": ["blaze_wolf"],
        "trigger": ["blaze wolf, caves and critters"],
    },
    "tala_(fluff-kevlar)": {
        "character": ["tala_(fluff-kevlar)"],
        "trigger": ["tala \\(fluff-kevlar\\), new year"],
    },
    "rinka_eya": {"character": ["rinka_eya"], "trigger": ["rinka eya, pepsi"]},
    "triple_d_(101_dalmatians)": {
        "character": ["triple_d_(101_dalmatians)"],
        "trigger": ["triple d \\(101 dalmatians\\), disney"],
    },
    "cherri_bomb_(hazbin_hotel)": {
        "character": ["cherri_bomb_(hazbin_hotel)"],
        "trigger": ["cherri bomb \\(hazbin hotel\\), hazbin hotel"],
    },
    "screw_(character)": {
        "character": ["screw_(character)"],
        "trigger": ["screw \\(character\\), my little pony"],
    },
    "small_norm": {
        "character": ["small_norm"],
        "trigger": ["small norm, crash bandicoot \\(series\\)"],
    },
    "judgement_(helltaker)": {
        "character": ["judgement_(helltaker)"],
        "trigger": ["judgement \\(helltaker\\), helltaker"],
    },
    "havoc_(tatsuchan18)": {
        "character": ["havoc_(tatsuchan18)"],
        "trigger": ["havoc \\(tatsuchan18\\), patreon"],
    },
    "giansar": {"character": ["giansar"], "trigger": ["giansar, lifewonders"]},
    "strych_(sincastermon)": {
        "character": ["strych_(sincastermon)"],
        "trigger": ["strych \\(sincastermon\\), ninja kiwi"],
    },
    "darkmon_(ryodramon)": {
        "character": ["darkmon_(ryodramon)"],
        "trigger": ["darkmon \\(ryodramon\\), digimon"],
    },
    "joshua_(zenthetiger)": {
        "character": ["joshua_(zenthetiger)"],
        "trigger": ["joshua \\(zenthetiger\\), nintendo"],
    },
    "amy_squirrel": {
        "character": ["amy_squirrel"],
        "trigger": ["amy squirrel, sabrina online"],
    },
    "mai_shiranui": {
        "character": ["mai_shiranui"],
        "trigger": ["mai shiranui, fatal fury"],
    },
    "keroro": {"character": ["keroro"], "trigger": ["keroro, sgt. frog"]},
    "kitty_(kimba)": {
        "character": ["kitty_(kimba)"],
        "trigger": ["kitty \\(kimba\\), osamu tezuka"],
    },
    "exile_(road_rovers)": {
        "character": ["exile_(road_rovers)"],
        "trigger": ["exile \\(road rovers\\), road rovers"],
    },
    "blue-eyes_white_dragon": {
        "character": ["blue-eyes_white_dragon"],
        "trigger": ["blue-eyes white dragon, yu-gi-oh!"],
    },
    "miura": {"character": ["miura"], "trigger": ["miura, ever oasis"]},
    "fvorte_(character)": {
        "character": ["fvorte_(character)"],
        "trigger": ["fvorte \\(character\\), mythology"],
    },
    "chiquitita_(shining)": {
        "character": ["chiquitita_(shining)"],
        "trigger": ["chiquitita \\(shining\\), sega"],
    },
    "reggie_(tokifuji)": {
        "character": ["reggie_(tokifuji)"],
        "trigger": ["reggie \\(tokifuji\\), patreon"],
    },
    "brush_(benju)": {
        "character": ["brush_(benju)"],
        "trigger": ["brush \\(benju\\), snapchat"],
    },
    "jonathan_stalizburg": {
        "character": ["jonathan_stalizburg"],
        "trigger": ["jonathan stalizburg, mother's day"],
    },
    "greta_(mlp)": {
        "character": ["greta_(mlp)"],
        "trigger": ["greta \\(mlp\\), my little pony"],
    },
    "firelander": {
        "character": ["firelander"],
        "trigger": ["firelander, my little pony"],
    },
    "boulder_(mlp)": {
        "character": ["boulder_(mlp)"],
        "trigger": ["boulder \\(mlp\\), my little pony"],
    },
    "noxy_(equinox)": {
        "character": ["noxy_(equinox)"],
        "trigger": ["noxy \\(equinox\\), my little pony"],
    },
    "stunbun": {"character": ["stunbun"], "trigger": ["stunbun, mythology"]},
    "kardukk": {"character": ["kardukk"], "trigger": ["kardukk, mythology"]},
    "wilson_(brogulls)": {
        "character": ["wilson_(brogulls)"],
        "trigger": ["wilson \\(brogulls\\), brogulls"],
    },
    "swissy": {"character": ["swissy"], "trigger": ["swissy, mythology"]},
    "mercy_(suelix)": {
        "character": ["mercy_(suelix)"],
        "trigger": ["mercy \\(suelix\\), tumblr"],
    },
    "misu_nox": {"character": ["misu_nox"], "trigger": ["misu nox, pokemon"]},
    "maya_henderson": {
        "character": ["maya_henderson"],
        "trigger": ["maya henderson, pokemon"],
    },
    "kindle_fae": {"character": ["kindle_fae"], "trigger": ["kindle fae, pokemon"]},
    "demino_(deminothedragon)": {
        "character": ["demino_(deminothedragon)"],
        "trigger": ["demino \\(deminothedragon\\), mythology"],
    },
    "srriz": {"character": ["srriz"], "trigger": ["srriz, srriz adventure"]},
    "nino_inukai": {
        "character": ["nino_inukai"],
        "trigger": ["nino inukai, christmas"],
    },
    "fenrir_(tatsuchan18)": {
        "character": ["fenrir_(tatsuchan18)"],
        "trigger": ["fenrir \\(tatsuchan18\\), game boy"],
    },
    "crome": {"character": ["crome"], "trigger": ["crome, mythology"]},
    "gtskunkrat_(character)": {
        "character": ["gtskunkrat_(character)"],
        "trigger": ["gtskunkrat \\(character\\), mythology"],
    },
    "rusteh_(sharkbum)": {
        "character": ["rusteh_(sharkbum)"],
        "trigger": ["rusteh \\(sharkbum\\), mythology"],
    },
    "bulma": {"character": ["bulma"], "trigger": ["bulma, dragon ball"]},
    "linahusky": {"character": ["linahusky"], "trigger": ["linahusky, mythology"]},
    "reis": {"character": ["reis"], "trigger": ["reis, unconditional \\(comic\\)"]},
    "betilla": {"character": ["betilla"], "trigger": ["betilla, ubisoft"]},
    "toucan_sam": {"character": ["toucan_sam"], "trigger": ["toucan sam, froot loops"]},
    "tiffy_cheesecake": {
        "character": ["tiffy_cheesecake"],
        "trigger": ["tiffy cheesecake, halloween"],
    },
    "w'rose_radiuju": {
        "character": ["w'rose_radiuju"],
        "trigger": ["w'rose radiuju, square enix"],
    },
    "jace_(gasaraki2007)": {
        "character": ["jace_(gasaraki2007)"],
        "trigger": ["jace \\(gasaraki2007\\), pokemon"],
    },
    "pokemon_go_trainer": {
        "character": ["pokemon_go_trainer"],
        "trigger": ["pokemon go trainer, pokemon"],
    },
    "katxlogan": {"character": ["katxlogan"], "trigger": ["katxlogan, nintendo"]},
    "corinoch": {"character": ["corinoch"], "trigger": ["corinoch, christmas"]},
    "kuehiko_roshihara": {
        "character": ["kuehiko_roshihara"],
        "trigger": ["kuehiko roshihara, working buddies!"],
    },
    "oscar_peltzer": {
        "character": ["oscar_peltzer"],
        "trigger": ["oscar peltzer, cartoon network"],
    },
    "kate_(jakethegoat)": {
        "character": ["kate_(jakethegoat)"],
        "trigger": ["kate \\(jakethegoat\\), mythology"],
    },
    "eugeniyburnt_(character)": {
        "character": ["eugeniyburnt_(character)"],
        "trigger": ["eugeniyburnt \\(character\\), twokinds"],
    },
    "mabel_(1-upclock)": {
        "character": ["mabel_(1-upclock)"],
        "trigger": ["mabel \\(1-upclock\\), pokemon"],
    },
    "syntia": {"character": ["syntia"], "trigger": ["syntia, blender \\(software\\)"]},
    "alvano_amala": {
        "character": ["alvano_amala"],
        "trigger": ["alvano amala, mythology"],
    },
    "martlet_(undertale_yellow)": {
        "character": ["martlet_(undertale_yellow)"],
        "trigger": ["martlet \\(undertale yellow\\), undertale yellow"],
    },
    "cabo_pompon_(unicorn_wars)": {
        "character": ["cabo_pompon_(unicorn_wars)"],
        "trigger": ["cabo pompon \\(unicorn wars\\), unicorn wars"],
    },
    "xpray_(character)": {
        "character": ["xpray_(character)"],
        "trigger": ["xpray \\(character\\), mythology"],
    },
    "zilla": {"character": ["zilla"], "trigger": ["zilla, godzilla \\(series\\)"]},
    "marvin_the_martian": {
        "character": ["marvin_the_martian"],
        "trigger": ["marvin the martian, looney tunes"],
    },
    "supergirl": {"character": ["supergirl"], "trigger": ["supergirl, dc comics"]},
    "scanty_daemon": {
        "character": ["scanty_daemon"],
        "trigger": ["scanty daemon, panty and stocking with garterbelt"],
    },
    "riven_(lol)": {
        "character": ["riven_(lol)"],
        "trigger": ["riven \\(lol\\), riot games"],
    },
    "sorrel": {"character": ["sorrel"], "trigger": ["sorrel, dragon ball"]},
    "meeka_rose": {
        "character": ["meeka_rose"],
        "trigger": ["meeka rose, tale of tails"],
    },
    "muriat": {"character": ["muriat"], "trigger": ["muriat, mythology"]},
    "leila_snowpaw": {
        "character": ["leila_snowpaw"],
        "trigger": ["leila snowpaw, mythology"],
    },
    "zera_(titsunekitsune)": {
        "character": ["zera_(titsunekitsune)"],
        "trigger": ["zera \\(titsunekitsune\\), warcraft"],
    },
    "jack_(zoophobia)": {
        "character": ["jack_(zoophobia)"],
        "trigger": ["jack \\(zoophobia\\), zoophobia"],
    },
    "sukimi_(hataraki)": {
        "character": ["sukimi_(hataraki)"],
        "trigger": ["sukimi \\(hataraki\\), greek mythology"],
    },
    "jasper_(steven_universe)": {
        "character": ["jasper_(steven_universe)"],
        "trigger": ["jasper \\(steven universe\\), cartoon network"],
    },
    "shadow_blue_(cloppermania)": {
        "character": ["shadow_blue_(cloppermania)"],
        "trigger": ["shadow blue \\(cloppermania\\), my little pony"],
    },
    "helbaa_(smutbooru)": {
        "character": ["helbaa_(smutbooru)"],
        "trigger": ["helbaa \\(smutbooru\\), disney"],
    },
    "chester_(bunnicula)": {
        "character": ["chester_(bunnicula)"],
        "trigger": ["chester \\(bunnicula\\), bunnicula \\(series\\)"],
    },
    "entaros_(character)": {
        "character": ["entaros_(character)"],
        "trigger": ["entaros \\(character\\), fortunate mixup"],
    },
    "grey_wolf_(kemono_friends)": {
        "character": ["grey_wolf_(kemono_friends)"],
        "trigger": ["grey wolf \\(kemono friends\\), kemono friends"],
    },
    "chaoz_(chaozdesignz)": {
        "character": ["chaoz_(chaozdesignz)"],
        "trigger": ["chaoz \\(chaozdesignz\\), darkstalkers"],
    },
    "daniel_(hladilnik)": {
        "character": ["daniel_(hladilnik)"],
        "trigger": ["daniel \\(hladilnik\\), toyota corolla"],
    },
    "aiden_laninga": {
        "character": ["aiden_laninga"],
        "trigger": ["aiden laninga, pokemon"],
    },
    "drift_(fortnite)": {
        "character": ["drift_(fortnite)"],
        "trigger": ["drift \\(fortnite\\), fortnite"],
    },
    "cricket_talot": {
        "character": ["cricket_talot"],
        "trigger": ["cricket talot, wizards of the coast"],
    },
    "pan_(sxfpantera)": {
        "character": ["pan_(sxfpantera)"],
        "trigger": ["pan \\(sxfpantera\\), mythology"],
    },
    "skarlett_cynder": {
        "character": ["skarlett_cynder"],
        "trigger": ["skarlett cynder, mythology"],
    },
    "khayen_(character)": {
        "character": ["khayen_(character)"],
        "trigger": ["khayen \\(character\\), mythology"],
    },
    "opal_(al_gx)": {
        "character": ["opal_(al_gx)"],
        "trigger": ["opal \\(al gx\\), pokemon"],
    },
    "nameless_(arbuzbudesh)": {
        "character": ["nameless_(arbuzbudesh)"],
        "trigger": ["nameless \\(arbuzbudesh\\)"],
    },
    "chris_chan": {
        "character": ["chris_chan"],
        "trigger": ["chris chan, sonichu \\(series\\)"],
    },
    "langdon_marston": {
        "character": ["langdon_marston"],
        "trigger": ["langdon marston, nintendo"],
    },
    "ann_gora": {"character": ["ann_gora"], "trigger": ["ann gora, swat kats"]},
    "kratos": {
        "character": ["kratos"],
        "trigger": ["kratos, sony interactive entertainment"],
    },
    "big_nug": {"character": ["big_nug"], "trigger": ["big nug, febreze"]},
    "daisy_(bcb)": {
        "character": ["daisy_(bcb)"],
        "trigger": ["daisy \\(bcb\\), bittersweet candy bowl"],
    },
    "morgana_(lol)": {
        "character": ["morgana_(lol)"],
        "trigger": ["morgana \\(lol\\), riot games"],
    },
    "luka_(monster_girl_quest)": {
        "character": ["luka_(monster_girl_quest)"],
        "trigger": ["luka \\(monster girl quest\\), monster girl quest"],
    },
    "doggylaw": {"character": ["doggylaw"], "trigger": ["doggylaw, mythology"]},
    "meowser": {"character": ["meowser"], "trigger": ["meowser, mario bros"]},
    "betsibi": {"character": ["betsibi"], "trigger": ["betsibi, nintendo"]},
    "donk_sis_(hladilnik)": {
        "character": ["donk_sis_(hladilnik)"],
        "trigger": ["donk sis \\(hladilnik\\), filthy frank"],
    },
    "sweet_voltage": {
        "character": ["sweet_voltage"],
        "trigger": ["sweet voltage, my little pony"],
    },
    "huntress_(risk_of_rain)": {
        "character": ["huntress_(risk_of_rain)"],
        "trigger": ["huntress \\(risk of rain\\), risk of rain"],
    },
    "filly_anon": {
        "character": ["filly_anon"],
        "trigger": ["filly anon, my little pony"],
    },
    "yule_(tas)": {
        "character": ["yule_(tas)"],
        "trigger": ["yule \\(tas\\), lifewonders"],
    },
    "lyre_belladonna": {
        "character": ["lyre_belladonna"],
        "trigger": ["lyre belladonna, pokemon"],
    },
    "kaz_mercais": {"character": ["kaz_mercais"], "trigger": ["kaz mercais, pokemon"]},
    "appledectomy": {
        "character": ["appledectomy"],
        "trigger": ["appledectomy, disney"],
    },
    "darius_davis": {"character": ["darius_davis"], "trigger": ["darius davis, honda"]},
    "daniela_idril": {
        "character": ["daniela_idril"],
        "trigger": ["daniela idril, transisters"],
    },
    "caitlyn_(swordfox)": {
        "character": ["caitlyn_(swordfox)"],
        "trigger": ["caitlyn \\(swordfox\\), pokemon"],
    },
    "cooper_(scratch21)": {
        "character": ["cooper_(scratch21)"],
        "trigger": ["cooper \\(scratch21\\), scratch21"],
    },
    "dolph_(fortnite)": {
        "character": ["dolph_(fortnite)"],
        "trigger": ["dolph \\(fortnite\\), fortnite"],
    },
    "wao_(e-zoid)": {
        "character": ["wao_(e-zoid)"],
        "trigger": ["wao \\(e-zoid\\), mythology"],
    },
    "zee_(abz)": {"character": ["zee_(abz)"], "trigger": ["zee \\(abz\\), abz"]},
    "erraz_(group17)": {
        "character": ["erraz_(group17)"],
        "trigger": ["erraz \\(group17\\), mythology"],
    },
    "asterius_(hades)": {
        "character": ["asterius_(hades)"],
        "trigger": ["asterius \\(hades\\), hades \\(game\\)"],
    },
    "kincade": {"character": ["kincade"], "trigger": ["kincade, furafterdark"]},
    "shrek_(character)": {
        "character": ["shrek_(character)"],
        "trigger": ["shrek \\(character\\), shrek \\(series\\)"],
    },
    "miles_lionheart": {
        "character": ["miles_lionheart"],
        "trigger": ["miles lionheart, las lindas"],
    },
    "chatot_(eotds)": {
        "character": ["chatot_(eotds)"],
        "trigger": ["chatot \\(eotds\\), pokemon mystery dungeon"],
    },
    "sora_(trias)": {
        "character": ["sora_(trias)"],
        "trigger": ["sora \\(trias\\), dinosaurs inc."],
    },
    "crossbreed_priscilla": {
        "character": ["crossbreed_priscilla"],
        "trigger": ["crossbreed priscilla, fromsoftware"],
    },
    "sly_asakura": {
        "character": ["sly_asakura"],
        "trigger": ["sly asakura, electricfox777"],
    },
    "soria": {"character": ["soria"], "trigger": ["soria, source filmmaker"]},
    "genji": {"character": ["genji"], "trigger": ["genji, digimon"]},
    "queen_elsa_(frozen)": {
        "character": ["queen_elsa_(frozen)"],
        "trigger": ["queen elsa \\(frozen\\), disney"],
    },
    "ruby_rose": {"character": ["ruby_rose"], "trigger": ["ruby rose, rwby"]},
    "arylon_lovire": {
        "character": ["arylon_lovire"],
        "trigger": ["arylon lovire, square enix"],
    },
    "maladash": {"character": ["maladash"], "trigger": ["maladash, starbound"]},
    "sora_(sorafoxyteils)": {
        "character": ["sora_(sorafoxyteils)"],
        "trigger": ["sora \\(sorafoxyteils\\), mythology"],
    },
    "honey_badger_(zootopia)": {
        "character": ["honey_badger_(zootopia)"],
        "trigger": ["honey badger \\(zootopia\\), disney"],
    },
    "director_ton": {
        "character": ["director_ton"],
        "trigger": ["director ton, sanrio"],
    },
    "danika_(wolflady)": {
        "character": ["danika_(wolflady)"],
        "trigger": ["danika \\(wolflady\\), nintendo"],
    },
    "moccha_(abluedeer)": {
        "character": ["moccha_(abluedeer)"],
        "trigger": ["moccha \\(abluedeer\\), moon lace"],
    },
    "zartersus": {"character": ["zartersus"], "trigger": ["zartersus, mythology"]},
    "maddie_flour": {
        "character": ["maddie_flour"],
        "trigger": ["maddie flour, disney"],
    },
    "kevin_(ac_stuart)": {
        "character": ["kevin_(ac_stuart)"],
        "trigger": ["kevin \\(ac stuart\\), awoo \\(ac stuart\\)"],
    },
    "rocco_(zoohomme)": {
        "character": ["rocco_(zoohomme)"],
        "trigger": ["rocco \\(zoohomme\\), zoohomme"],
    },
    "kidden_eksis": {
        "character": ["kidden_eksis"],
        "trigger": ["kidden eksis, mythology"],
    },
    "meowscles_(shadow)": {
        "character": ["meowscles_(shadow)"],
        "trigger": ["meowscles \\(shadow\\), fortnite"],
    },
    "sophia_(xxsparcoxx)": {
        "character": ["sophia_(xxsparcoxx)"],
        "trigger": ["sophia \\(xxsparcoxx\\), resident evil"],
    },
    "sora_(tehkey)": {
        "character": ["sora_(tehkey)"],
        "trigger": ["sora \\(tehkey\\), nintendo"],
    },
    "opaline_(mlp)": {
        "character": ["opaline_(mlp)"],
        "trigger": ["opaline \\(mlp\\), mlp g5"],
    },
    "zyneru_(character)": {
        "character": ["zyneru_(character)"],
        "trigger": ["zyneru \\(character\\), new year"],
    },
    "sall_(mistyy_draws)": {
        "character": ["sall_(mistyy_draws)"],
        "trigger": ["sall \\(mistyy draws\\), nintendo"],
    },
    "freddy_krueger": {
        "character": ["freddy_krueger"],
        "trigger": ["freddy krueger, nightmare on elm street"],
    },
    "sasha_gothica": {
        "character": ["sasha_gothica"],
        "trigger": ["sasha gothica, brave new world \\(style wager\\)"],
    },
    "janice_carter": {
        "character": ["janice_carter"],
        "trigger": ["janice carter, nhl"],
    },
    "chen_stormstout": {
        "character": ["chen_stormstout"],
        "trigger": ["chen stormstout, warcraft"],
    },
    "impreza": {
        "character": ["impreza"],
        "trigger": ["impreza, blender \\(software\\)"],
    },
    "chiu": {"character": ["chiu"], "trigger": ["chiu, dragon quest"]},
    "rainier_(rain-yatsu)": {
        "character": ["rainier_(rain-yatsu)"],
        "trigger": ["rainier \\(rain-yatsu\\), seattle fur"],
    },
    "nyanta": {"character": ["nyanta"], "trigger": ["nyanta, log horizon"]},
    "cordite": {"character": ["cordite"], "trigger": ["cordite, mythology"]},
    "emmitt_otterton": {
        "character": ["emmitt_otterton"],
        "trigger": ["emmitt otterton, disney"],
    },
    "proby": {"character": ["proby"], "trigger": ["proby, linklynx"]},
    "tionishia_(monster_musume)": {
        "character": ["tionishia_(monster_musume)"],
        "trigger": ["tionishia \\(monster musume\\), monster musume"],
    },
    "manizu": {"character": ["manizu"], "trigger": ["manizu, tenga"]},
    "fuzeyeen": {"character": ["fuzeyeen"], "trigger": ["fuzeyeen, nintendo"]},
    "alilkira": {"character": ["alilkira"], "trigger": ["alilkira, warfare machine"]},
    "pharynx_(mlp)": {
        "character": ["pharynx_(mlp)"],
        "trigger": ["pharynx \\(mlp\\), my little pony"],
    },
    "olivia_hart": {
        "character": ["olivia_hart"],
        "trigger": ["olivia hart, book of lust"],
    },
    "grim_matchstick": {
        "character": ["grim_matchstick"],
        "trigger": ["grim matchstick, cuphead \\(game\\)"],
    },
    "caesar_(anglo)": {
        "character": ["caesar_(anglo)"],
        "trigger": ["caesar \\(anglo\\), pokemon"],
    },
    "cinder_glow_(mlp)": {
        "character": ["cinder_glow_(mlp)"],
        "trigger": ["cinder glow \\(mlp\\), my little pony"],
    },
    "marshmallow_fluff_(character)": {
        "character": ["marshmallow_fluff_(character)"],
        "trigger": ["marshmallow fluff \\(character\\), mythology"],
    },
    "neon_mitsumi": {
        "character": ["neon_mitsumi"],
        "trigger": ["neon mitsumi, twokinds"],
    },
    "moses_(samur_shalem)": {
        "character": ["moses_(samur_shalem)"],
        "trigger": ["moses \\(samur shalem\\), disney"],
    },
    "novus_(kitfox-crimson)": {
        "character": ["novus_(kitfox-crimson)"],
        "trigger": ["novus \\(kitfox-crimson\\), stolen generation"],
    },
    "nicky_(abfmh)": {
        "character": ["nicky_(abfmh)"],
        "trigger": ["nicky \\(abfmh\\), furaffinity"],
    },
    "alicia_pris": {
        "character": ["alicia_pris"],
        "trigger": ["alicia pris, little tail bronx"],
    },
    "yoruichi_shihoin": {
        "character": ["yoruichi_shihoin"],
        "trigger": ["yoruichi shihoin, bleach \\(series\\)"],
    },
    "marilyn_monroe": {
        "character": ["marilyn_monroe"],
        "trigger": ["marilyn monroe, the seven year itch"],
    },
    "firefly_(pre-g4)": {
        "character": ["firefly_(pre-g4)"],
        "trigger": ["firefly \\(pre-g4\\), my little pony"],
    },
    "sonichu_(character)": {
        "character": ["sonichu_(character)"],
        "trigger": ["sonichu \\(character\\), sonichu \\(series\\)"],
    },
    "thundergrey": {
        "character": ["thundergrey"],
        "trigger": ["thundergrey, mythology"],
    },
    "inoby_(character)": {
        "character": ["inoby_(character)"],
        "trigger": ["inoby \\(character\\), mythology"],
    },
    "peeka_(mario)": {
        "character": ["peeka_(mario)"],
        "trigger": ["peeka \\(mario\\), mario bros"],
    },
    "xianos": {"character": ["xianos"], "trigger": ["xianos, mythology"]},
    "pirate_leader_tetra": {
        "character": ["pirate_leader_tetra"],
        "trigger": ["pirate leader tetra, the legend of zelda"],
    },
    "sigma_x_(character)": {
        "character": ["sigma_x_(character)"],
        "trigger": ["sigma x \\(character\\), patreon"],
    },
    "terezi_pyrope": {
        "character": ["terezi_pyrope"],
        "trigger": ["terezi pyrope, homestuck"],
    },
    "dodger_(disney)": {
        "character": ["dodger_(disney)"],
        "trigger": ["dodger \\(disney\\), disney"],
    },
    "gradie": {
        "character": ["gradie"],
        "trigger": ["gradie, ludwig bullworth jackson \\(copyright\\)"],
    },
    "gabrielle_(legend_of_queen_opala)": {
        "character": ["gabrielle_(legend_of_queen_opala)"],
        "trigger": ["gabrielle \\(legend of queen opala\\), legend of queen opala"],
    },
    "bry": {"character": ["bry"], "trigger": ["bry, christmas"]},
    "styx_(jelomaus)": {
        "character": ["styx_(jelomaus)"],
        "trigger": ["styx \\(jelomaus\\), mythology"],
    },
    "snowcheetah": {
        "character": ["snowcheetah"],
        "trigger": ["snowcheetah, mythology"],
    },
    "alfred_(umpherio)": {
        "character": ["alfred_(umpherio)"],
        "trigger": ["alfred \\(umpherio\\), pokemon"],
    },
    "cato_(peritian)": {
        "character": ["cato_(peritian)"],
        "trigger": ["cato \\(peritian\\), adam lambert"],
    },
    "racket_rhine": {
        "character": ["racket_rhine"],
        "trigger": ["racket rhine, my little pony"],
    },
    "sasha_sweets": {
        "character": ["sasha_sweets"],
        "trigger": ["sasha sweets, nintendo"],
    },
    "tania_tlacuache": {
        "character": ["tania_tlacuache"],
        "trigger": ["tania tlacuache, twitch.tv"],
    },
    "soul-silver-dragon_(character)": {
        "character": ["soul-silver-dragon_(character)"],
        "trigger": ["soul-silver-dragon \\(character\\), mythology"],
    },
    "anna_(study_partners)": {
        "character": ["anna_(study_partners)"],
        "trigger": ["anna \\(study partners\\), study partners"],
    },
    "famir_(character)": {
        "character": ["famir_(character)"],
        "trigger": ["famir \\(character\\), mythology"],
    },
    "felix_(striped_sins)": {
        "character": ["felix_(striped_sins)"],
        "trigger": ["felix \\(striped sins\\), striped sins"],
    },
    "thomas_(zourik)": {
        "character": ["thomas_(zourik)"],
        "trigger": ["thomas \\(zourik\\), mythology"],
    },
    "sploot_(unknownspy)": {
        "character": ["sploot_(unknownspy)"],
        "trigger": ["sploot \\(unknownspy\\), undertale \\(series\\)"],
    },
    "alchemist_(bloons)": {
        "character": ["alchemist_(bloons)"],
        "trigger": ["alchemist \\(bloons\\), ninja kiwi"],
    },
    "azura_(aimpunch)": {
        "character": ["azura_(aimpunch)"],
        "trigger": ["azura \\(aimpunch\\), fortnite"],
    },
    "daena": {"character": ["daena"], "trigger": ["daena, square enix"]},
    "rick2tails": {
        "character": ["rick2tails"],
        "trigger": ["rick2tails, sonic the hedgehog \\(series\\)"],
    },
    "ethan_bedlam": {
        "character": ["ethan_bedlam"],
        "trigger": ["ethan bedlam, koikatsu \\(game\\)"],
    },
    "sir_fratley": {
        "character": ["sir_fratley"],
        "trigger": ["sir fratley, final fantasy ix"],
    },
    "lyra_(pokemon)": {
        "character": ["lyra_(pokemon)"],
        "trigger": ["lyra \\(pokemon\\), pokemon"],
    },
    "rev_runner": {
        "character": ["rev_runner"],
        "trigger": ["rev runner, loonatics unleashed"],
    },
    "nina_tucker": {
        "character": ["nina_tucker"],
        "trigger": ["nina tucker, fullmetal alchemist"],
    },
    "android_18": {"character": ["android_18"], "trigger": ["android 18, dragon ball"]},
    "nan_(nq)": {"character": ["nan_(nq)"], "trigger": ["nan \\(nq\\), nan quest"]},
    "domo_(ben300)": {
        "character": ["domo_(ben300)"],
        "trigger": ["domo \\(ben300\\), mythology"],
    },
    "flam_(mlp)": {
        "character": ["flam_(mlp)"],
        "trigger": ["flam \\(mlp\\), my little pony"],
    },
    "faunoiphilia": {
        "character": ["faunoiphilia"],
        "trigger": ["faunoiphilia, mythology"],
    },
    "ebony_marionette_georg": {
        "character": ["ebony_marionette_georg"],
        "trigger": ["ebony marionette georg, sonic the hedgehog \\(series\\)"],
    },
    "bamwuff": {"character": ["bamwuff"], "trigger": ["bamwuff, mythology"]},
    "sorez_(mastersorez)": {
        "character": ["sorez_(mastersorez)"],
        "trigger": ["sorez \\(mastersorez\\), mythology"],
    },
    "latch": {"character": ["latch"], "trigger": ["latch, lethal league"]},
    "bonita_(gaturo)": {
        "character": ["bonita_(gaturo)"],
        "trigger": ["bonita \\(gaturo\\), nintendo ds family"],
    },
    "sydney_swamp_(vimhomeless)": {
        "character": ["sydney_swamp_(vimhomeless)"],
        "trigger": ["sydney swamp \\(vimhomeless\\), mythology"],
    },
    "napoleon_(underscore-b)": {
        "character": ["napoleon_(underscore-b)"],
        "trigger": ["napoleon \\(underscore-b\\), mythology"],
    },
    "fuckboy": {"character": ["fuckboy"], "trigger": ["fuckboy, pokemon"]},
    "kai_(kaibun)": {
        "character": ["kai_(kaibun)"],
        "trigger": ["kai \\(kaibun\\), nintendo"],
    },
    "ren_amamiya": {"character": ["ren_amamiya"], "trigger": ["ren amamiya, sega"]},
    "connor_the_gaomon": {
        "character": ["connor_the_gaomon"],
        "trigger": ["connor the gaomon, digimon"],
    },
    "delia_(anglo)": {
        "character": ["delia_(anglo)"],
        "trigger": ["delia \\(anglo\\), pokemon"],
    },
    "spitfire_(hideki_kaneda)": {
        "character": ["spitfire_(hideki_kaneda)"],
        "trigger": ["spitfire \\(hideki kaneda\\), supermarine spitfire"],
    },
    "willow_wisp": {"character": ["willow_wisp"], "trigger": ["willow wisp, twokinds"]},
    "rabbit_(petruz)": {
        "character": ["rabbit_(petruz)"],
        "trigger": ["rabbit \\(petruz\\), source filmmaker"],
    },
    "thomas_ii_(zer0rebel4)": {
        "character": ["thomas_ii_(zer0rebel4)"],
        "trigger": ["thomas ii \\(zer0rebel4\\), mythology"],
    },
    "trip_the_sungazer": {
        "character": ["trip_the_sungazer"],
        "trigger": ["trip the sungazer, sonic the hedgehog \\(series\\)"],
    },
    "dickbutt": {"character": ["dickbutt"], "trigger": ["dickbutt, nintendo"]},
    "flannery_(pokemon)": {
        "character": ["flannery_(pokemon)"],
        "trigger": ["flannery \\(pokemon\\), pokemon"],
    },
    "karkat_vantas": {
        "character": ["karkat_vantas"],
        "trigger": ["karkat vantas, homestuck"],
    },
    "rytlock_brimstone": {
        "character": ["rytlock_brimstone"],
        "trigger": ["rytlock brimstone, guild wars"],
    },
    "tiger_trace": {"character": ["tiger_trace"], "trigger": ["tiger trace, twokinds"]},
    "night_(nightfaux)": {
        "character": ["night_(nightfaux)"],
        "trigger": ["night \\(nightfaux\\), mythology"],
    },
    "vagus_(haychel)": {
        "character": ["vagus_(haychel)"],
        "trigger": ["vagus \\(haychel\\), pokemon"],
    },
    "snakehead404": {
        "character": ["snakehead404"],
        "trigger": ["snakehead404, alien \\(franchise\\)"],
    },
    "navos": {"character": ["navos"], "trigger": ["navos, mythology"]},
    "ladies_of_the_shade": {
        "character": ["ladies_of_the_shade"],
        "trigger": ["ladies of the shade, kung fu panda"],
    },
    "saffira_queen_of_dragons": {
        "character": ["saffira_queen_of_dragons"],
        "trigger": ["saffira queen of dragons, yu-gi-oh!"],
    },
    "yrel": {"character": ["yrel"], "trigger": ["yrel, warcraft"]},
    "magnus_(spyro)": {
        "character": ["magnus_(spyro)"],
        "trigger": ["magnus \\(spyro\\), spyro the dragon"],
    },
    "melody_(mellybyte)": {
        "character": ["melody_(mellybyte)"],
        "trigger": ["melody \\(mellybyte\\), mythology"],
    },
    "bau_husky": {
        "character": ["bau_husky"],
        "trigger": ["bau husky, gab \\(comic\\)"],
    },
    "bill_(hladilnik)": {
        "character": ["bill_(hladilnik)"],
        "trigger": ["bill \\(hladilnik\\), smith & wesson"],
    },
    "krisha_russel": {
        "character": ["krisha_russel"],
        "trigger": ["krisha russel, mythology"],
    },
    "sasha_(bunnybits)": {
        "character": ["sasha_(bunnybits)"],
        "trigger": ["sasha \\(bunnybits\\), nintendo"],
    },
    "sheila_(beastars)": {
        "character": ["sheila_(beastars)"],
        "trigger": ["sheila \\(beastars\\), beastars"],
    },
    "youngster_(pokemon_sword_and_shield)": {
        "character": ["youngster_(pokemon_sword_and_shield)"],
        "trigger": ["youngster \\(pokemon sword and shield\\), pokemon"],
    },
    "blaze_(agitype01)": {
        "character": ["blaze_(agitype01)"],
        "trigger": ["blaze \\(agitype01\\), pokemon"],
    },
    "kokkoro_(princess_connect!)": {
        "character": ["kokkoro_(princess_connect!)"],
        "trigger": ["kokkoro \\(princess connect!\\), cygames"],
    },
    "mathilda_(jamoart)": {
        "character": ["mathilda_(jamoart)"],
        "trigger": ["mathilda \\(jamoart\\), strip meme"],
    },
    "desire_(icma)": {
        "character": ["desire_(icma)"],
        "trigger": ["desire \\(icma\\), pokemon"],
    },
    "shijin": {
        "character": ["shijin"],
        "trigger": ["shijin, gamba no bouken \\(series\\)"],
    },
    "gusta_(gusta)": {
        "character": ["gusta_(gusta)"],
        "trigger": ["gusta \\(gusta\\), mythology"],
    },
    "paul_(zourik)": {
        "character": ["paul_(zourik)"],
        "trigger": ["paul \\(zourik\\), mythology"],
    },
    "ichi_inukai": {
        "character": ["ichi_inukai"],
        "trigger": ["ichi inukai, christmas"],
    },
    "akari_jamisson": {
        "character": ["akari_jamisson"],
        "trigger": ["akari jamisson, pokemon"],
    },
    "vayhl'ayne_(vahlyance)": {
        "character": ["vayhl'ayne_(vahlyance)"],
        "trigger": ["vayhl'ayne \\(vahlyance\\), monster hunter"],
    },
    "angela_(badgerben)": {
        "character": ["angela_(badgerben)"],
        "trigger": ["angela \\(badgerben\\), mythology"],
    },
    "cortana_(halo)": {
        "character": ["cortana_(halo)"],
        "trigger": ["cortana \\(halo\\), halo \\(series\\)"],
    },
    "lahla_(mario)": {
        "character": ["lahla_(mario)"],
        "trigger": ["lahla \\(mario\\), mario bros"],
    },
    "cocoa_(las_lindas)": {
        "character": ["cocoa_(las_lindas)"],
        "trigger": ["cocoa \\(las lindas\\), las lindas"],
    },
    "homer_simpson": {
        "character": ["homer_simpson"],
        "trigger": ["homer simpson, the simpsons"],
    },
    "optimus_prime": {
        "character": ["optimus_prime"],
        "trigger": ["optimus prime, takara tomy"],
    },
    "swizz_(swizzlestix)": {
        "character": ["swizz_(swizzlestix)"],
        "trigger": ["swizz \\(swizzlestix\\), mythology"],
    },
    "olivia_flaversham": {
        "character": ["olivia_flaversham"],
        "trigger": ["olivia flaversham, disney"],
    },
    "shyvana": {"character": ["shyvana"], "trigger": ["shyvana, riot games"]},
    "randall_boggs": {
        "character": ["randall_boggs"],
        "trigger": ["randall boggs, disney"],
    },
    "hellen_lockheart": {
        "character": ["hellen_lockheart"],
        "trigger": ["hellen lockheart, my little pony"],
    },
    "skull_grunt": {"character": ["skull_grunt"], "trigger": ["skull grunt, pokemon"]},
    "queen_novo_(mlp)": {
        "character": ["queen_novo_(mlp)"],
        "trigger": ["queen novo \\(mlp\\), my little pony"],
    },
    "seraphic_crimson": {
        "character": ["seraphic_crimson"],
        "trigger": ["seraphic crimson, mythology"],
    },
    "gosha_(beastars)": {
        "character": ["gosha_(beastars)"],
        "trigger": ["gosha \\(beastars\\), beastars"],
    },
    "sakura_d._lyall": {
        "character": ["sakura_d._lyall"],
        "trigger": ["sakura d. lyall, mythology"],
    },
    "sherb_(animal_crossing)": {
        "character": ["sherb_(animal_crossing)"],
        "trigger": ["sherb \\(animal crossing\\), animal crossing"],
    },
    "emerald_(ultilix)": {
        "character": ["emerald_(ultilix)"],
        "trigger": ["emerald \\(ultilix\\), mythology"],
    },
    "vixy_(hyilpi)": {
        "character": ["vixy_(hyilpi)"],
        "trigger": ["vixy \\(hyilpi\\), minecraft"],
    },
    "trevor_pride_(knotfunny)": {
        "character": ["trevor_pride_(knotfunny)"],
        "trigger": ["trevor pride \\(knotfunny\\), mythology"],
    },
    "sara_(phrannd)": {
        "character": ["sara_(phrannd)"],
        "trigger": ["sara \\(phrannd\\), mythology"],
    },
    "6-9": {"character": ["6-9"], "trigger": ["6-9, mythology"]},
    "eddie_(doginacafe)": {
        "character": ["eddie_(doginacafe)"],
        "trigger": ["eddie \\(doginacafe\\), lgbt history month"],
    },
    "akino_(kemokin_mania)": {
        "character": ["akino_(kemokin_mania)"],
        "trigger": ["akino \\(kemokin mania\\), chinese zodiac"],
    },
    "andrealphus_(helluva_boss)": {
        "character": ["andrealphus_(helluva_boss)"],
        "trigger": ["andrealphus \\(helluva boss\\), helluva boss"],
    },
    "pipa_(kitfox-crimson)": {
        "character": ["pipa_(kitfox-crimson)"],
        "trigger": ["pipa \\(kitfox-crimson\\), in our shadow"],
    },
    "james_(team_rocket)": {
        "character": ["james_(team_rocket)"],
        "trigger": ["james \\(team rocket\\), team rocket"],
    },
    "venus_de_milo_(tmnt)": {
        "character": ["venus_de_milo_(tmnt)"],
        "trigger": ["venus de milo \\(tmnt\\), teenage mutant ninja turtles"],
    },
    "brain_(inspector_gadget)": {
        "character": ["brain_(inspector_gadget)"],
        "trigger": ["brain \\(inspector gadget\\), inspector gadget \\(franchise\\)"],
    },
    "diego_(ice_age)": {
        "character": ["diego_(ice_age)"],
        "trigger": ["diego \\(ice age\\), ice age \\(series\\)"],
    },
    "xasyr": {"character": ["xasyr"], "trigger": ["xasyr, mythology"]},
    "ryan_moonshadow": {
        "character": ["ryan_moonshadow"],
        "trigger": ["ryan moonshadow, mythology"],
    },
    "pantherlily": {
        "character": ["pantherlily"],
        "trigger": ["pantherlily, fairy tail"],
    },
    "coco_nebulon": {
        "character": ["coco_nebulon"],
        "trigger": ["coco nebulon, awesomenauts"],
    },
    "ultima_(oc)": {
        "character": ["ultima_(oc)"],
        "trigger": ["ultima \\(oc\\), my little pony"],
    },
    "arun_(tokaido)": {
        "character": ["arun_(tokaido)"],
        "trigger": ["arun \\(tokaido\\), mythology"],
    },
    "calpain": {"character": ["calpain"], "trigger": ["calpain, my little pony"]},
    "alori_dawnstar": {
        "character": ["alori_dawnstar"],
        "trigger": ["alori dawnstar, warcraft"],
    },
    "mocha_(eikasianspire)": {
        "character": ["mocha_(eikasianspire)"],
        "trigger": ["mocha \\(eikasianspire\\), tumblr"],
    },
    "biobatz": {"character": ["biobatz"], "trigger": ["biobatz, mythology"]},
    "ruby_rustfeather_(nakuk)": {
        "character": ["ruby_rustfeather_(nakuk)"],
        "trigger": ["ruby rustfeather \\(nakuk\\), starbound"],
    },
    "iqua_kicks": {"character": ["iqua_kicks"], "trigger": ["iqua kicks, patreon"]},
    "twee": {"character": ["twee"], "trigger": ["twee, nintendo"]},
    "shani_(zummeng)": {
        "character": ["shani_(zummeng)"],
        "trigger": ["shani \\(zummeng\\), patreon"],
    },
    "ash_(g-h-)": {"character": ["ash_(g-h-)"], "trigger": ["ash \\(g-h-\\), pokemon"]},
    "ashley_(ashleyboi)": {
        "character": ["ashley_(ashleyboi)"],
        "trigger": ["ashley \\(ashleyboi\\), mythology"],
    },
    "klaus_(shakotanbunny)": {
        "character": ["klaus_(shakotanbunny)"],
        "trigger": ["klaus \\(shakotanbunny\\), my little pony"],
    },
    "christa_(rebeldragon101)": {
        "character": ["christa_(rebeldragon101)"],
        "trigger": ["christa \\(rebeldragon101\\), mythology"],
    },
    "samuel_ayers": {
        "character": ["samuel_ayers"],
        "trigger": ["samuel ayers, the smoke room"],
    },
    "erraz_sandwalker": {
        "character": ["erraz_sandwalker"],
        "trigger": ["erraz sandwalker, mythology"],
    },
    "cantio_(lawyerdog)": {
        "character": ["cantio_(lawyerdog)"],
        "trigger": ["cantio \\(lawyerdog\\), meme clothing"],
    },
    "lobo_(lobokosmico)": {
        "character": ["lobo_(lobokosmico)"],
        "trigger": ["lobo \\(lobokosmico\\), disney"],
    },
    "clover_(undertale_yellow)": {
        "character": ["clover_(undertale_yellow)"],
        "trigger": ["clover \\(undertale yellow\\), undertale yellow"],
    },
    "tamama": {"character": ["tamama"], "trigger": ["tamama, sgt. frog"]},
    "danza_(character)": {
        "character": ["danza_(character)"],
        "trigger": ["danza \\(character\\), mythology"],
    },
    "barney_the_dinosaur": {
        "character": ["barney_the_dinosaur"],
        "trigger": ["barney the dinosaur, barney and friends"],
    },
    "amon_(rukis)": {
        "character": ["amon_(rukis)"],
        "trigger": ["amon \\(rukis\\), red lantern"],
    },
    "iris_(pokemon)": {
        "character": ["iris_(pokemon)"],
        "trigger": ["iris \\(pokemon\\), pokemon"],
    },
    "sila_dione": {"character": ["sila_dione"], "trigger": ["sila dione, christmas"]},
    "lolori": {"character": ["lolori"], "trigger": ["lolori, mythology"]},
    "seljhet": {"character": ["seljhet"], "trigger": ["seljhet, mythology"]},
    "webber": {"character": ["webber"], "trigger": ["webber, don't starve"]},
    "dumbun": {"character": ["dumbun"], "trigger": ["dumbun, nintendo"]},
    "drew_dubsky": {
        "character": ["drew_dubsky"],
        "trigger": ["drew dubsky, mythology"],
    },
    "susie_(kirby)": {
        "character": ["susie_(kirby)"],
        "trigger": ["susie \\(kirby\\), kirby \\(series\\)"],
    },
    "forefox": {"character": ["forefox"], "trigger": ["forefox, pokemon"]},
    "shashe'_saramunra": {
        "character": ["shashe'_saramunra"],
        "trigger": ["shashe' saramunra, halo \\(series\\)"],
    },
    "spectre_phase_(oc)": {
        "character": ["spectre_phase_(oc)"],
        "trigger": ["spectre phase \\(oc\\), my little pony"],
    },
    "avi_(darkastray)": {
        "character": ["avi_(darkastray)"],
        "trigger": ["avi \\(darkastray\\), mythology"],
    },
    "paya": {"character": ["paya"], "trigger": ["paya, the legend of zelda"]},
    "quitela": {"character": ["quitela"], "trigger": ["quitela, dragon ball"]},
    "s.leech_(oc)": {
        "character": ["s.leech_(oc)"],
        "trigger": ["s.leech \\(oc\\), my little pony"],
    },
    "lazuli_delarosa": {
        "character": ["lazuli_delarosa"],
        "trigger": ["lazuli delarosa, mythology"],
    },
    "qibli_(wof)": {
        "character": ["qibli_(wof)"],
        "trigger": ["qibli \\(wof\\), mythology"],
    },
    "flay_(wingedwilly)": {
        "character": ["flay_(wingedwilly)"],
        "trigger": ["flay \\(wingedwilly\\), mythology"],
    },
    "fiona_fawnbags_(dullvivid)": {
        "character": ["fiona_fawnbags_(dullvivid)"],
        "trigger": ["fiona fawnbags \\(dullvivid\\), source filmmaker"],
    },
    "agata_(beastars)": {
        "character": ["agata_(beastars)"],
        "trigger": ["agata \\(beastars\\), beastars"],
    },
    "roy_(beez)": {"character": ["roy_(beez)"], "trigger": ["roy \\(beez\\), patreon"]},
    "lolly_(butterscotchlollipop)": {
        "character": ["lolly_(butterscotchlollipop)"],
        "trigger": ["lolly \\(butterscotchlollipop\\), christmas"],
    },
    "stra_(icma)": {
        "character": ["stra_(icma)"],
        "trigger": ["stra \\(icma\\), pmd: icma"],
    },
    "frankie_(extremedash)": {
        "character": ["frankie_(extremedash)"],
        "trigger": ["frankie \\(extremedash\\), dog knight rpg"],
    },
    "huckle": {"character": ["huckle"], "trigger": ["huckle, lifewonders"]},
    "lee_(arknights)": {
        "character": ["lee_(arknights)"],
        "trigger": ["lee \\(arknights\\), studio montagne"],
    },
    "glori_gamebird": {
        "character": ["glori_gamebird"],
        "trigger": ["glori gamebird, mythology"],
    },
    "pedobear": {"character": ["pedobear"], "trigger": ["pedobear, pokemon"]},
    "rei_ayanami": {
        "character": ["rei_ayanami"],
        "trigger": ["rei ayanami, neon genesis evangelion"],
    },
    "dawn_(jeremy_bernal)": {
        "character": ["dawn_(jeremy_bernal)"],
        "trigger": ["dawn \\(jeremy bernal\\), patreon"],
    },
    "pecas_(freckles)": {
        "character": ["pecas_(freckles)"],
        "trigger": ["pecas \\(freckles\\), mythology"],
    },
    "cam_collins": {
        "character": ["cam_collins"],
        "trigger": ["cam collins, mythology"],
    },
    "marcus_mccloud": {
        "character": ["marcus_mccloud"],
        "trigger": ["marcus mccloud, star fox"],
    },
    "goldie_(animal_crossing)": {
        "character": ["goldie_(animal_crossing)"],
        "trigger": ["goldie \\(animal crossing\\), animal crossing"],
    },
    "scott_otter": {
        "character": ["scott_otter"],
        "trigger": ["scott otter, mythology"],
    },
    "maxamilion_the_fox": {
        "character": ["maxamilion_the_fox"],
        "trigger": ["maxamilion the fox, fire island entertainment"],
    },
    "curly_q": {"character": ["curly_q"], "trigger": ["curly q, pokemon"]},
    "scourge_(warriors)": {
        "character": ["scourge_(warriors)"],
        "trigger": ["scourge \\(warriors\\), warriors \\(book series\\)"],
    },
    "charlotte_(phurie)": {
        "character": ["charlotte_(phurie)"],
        "trigger": ["charlotte \\(phurie\\), my little pony"],
    },
    "alesia": {"character": ["alesia"], "trigger": ["alesia, mythology"]},
    "harbour_princess": {
        "character": ["harbour_princess"],
        "trigger": ["harbour princess, kantai collection"],
    },
    "wraith_(evolve)": {
        "character": ["wraith_(evolve)"],
        "trigger": ["wraith \\(evolve\\), evolve \\(copyright\\)"],
    },
    "violet_skies_(character)": {
        "character": ["violet_skies_(character)"],
        "trigger": ["violet skies \\(character\\), my little pony"],
    },
    "kess_(coffeechicken)": {
        "character": ["kess_(coffeechicken)"],
        "trigger": ["kess \\(coffeechicken\\), new year"],
    },
    "dogamy": {"character": ["dogamy"], "trigger": ["dogamy, undertale \\(series\\)"]},
    "electro_current_(oc)": {
        "character": ["electro_current_(oc)"],
        "trigger": ["electro current \\(oc\\), my little pony"],
    },
    "duo_(duolingo)": {
        "character": ["duo_(duolingo)"],
        "trigger": ["duo \\(duolingo\\), duolingo"],
    },
    "lee_the_kec": {"character": ["lee_the_kec"], "trigger": ["lee the kec, pokemon"]},
    "tabatha_(samoyena)": {
        "character": ["tabatha_(samoyena)"],
        "trigger": ["tabatha \\(samoyena\\), new year"],
    },
    "adhira_hale": {
        "character": ["adhira_hale"],
        "trigger": ["adhira hale, mythology"],
    },
    "ashley_(mewgle)": {
        "character": ["ashley_(mewgle)"],
        "trigger": ["ashley \\(mewgle\\), christmas"],
    },
    "letho_donovan": {
        "character": ["letho_donovan"],
        "trigger": ["letho donovan, the doors"],
    },
    "chaut": {"character": ["chaut"], "trigger": ["chaut, pokemon"]},
    "dal_(dal_your_pal)": {
        "character": ["dal_(dal_your_pal)"],
        "trigger": ["dal \\(dal your pal\\), mythology"],
    },
    "lumina_(stargazer)": {
        "character": ["lumina_(stargazer)"],
        "trigger": ["lumina \\(stargazer\\), christmas"],
    },
    "bryan_(zourik)": {
        "character": ["bryan_(zourik)"],
        "trigger": ["bryan \\(zourik\\), pokemon"],
    },
    "zander_(zhanbow)": {
        "character": ["zander_(zhanbow)"],
        "trigger": ["zander \\(zhanbow\\), mythology"],
    },
    "skye_(ratcha)": {
        "character": ["skye_(ratcha)"],
        "trigger": ["skye \\(ratcha\\), wizards of the coast"],
    },
    "wammawink": {"character": ["wammawink"], "trigger": ["wammawink, netflix"]},
    "rigbette_(benson_dancing)": {
        "character": ["rigbette_(benson_dancing)"],
        "trigger": ["rigbette \\(benson dancing\\), cartoon network"],
    },
    "haolde_(character)": {
        "character": ["haolde_(character)"],
        "trigger": ["haolde \\(character\\), blender \\(software\\)"],
    },
    "mutio": {"character": ["mutio"], "trigger": ["mutio, blue submarine no. 6"]},
    "pussy_noir": {
        "character": ["pussy_noir"],
        "trigger": ["pussy noir, furafterdark"],
    },
    "vinnie_(bmfm)": {
        "character": ["vinnie_(bmfm)"],
        "trigger": ["vinnie \\(bmfm\\), biker mice from mars"],
    },
    "yogi_bear_(character)": {
        "character": ["yogi_bear_(character)"],
        "trigger": ["yogi bear \\(character\\), yogi bear"],
    },
    "smurfette": {"character": ["smurfette"], "trigger": ["smurfette, the smurfs"]},
    "shendu": {"character": ["shendu"], "trigger": ["shendu, mythology"]},
    "hinata_hyuga": {
        "character": ["hinata_hyuga"],
        "trigger": ["hinata hyuga, naruto"],
    },
    "tephros": {"character": ["tephros"], "trigger": ["tephros, mythology"]},
    "nik_(sonicfox)": {
        "character": ["nik_(sonicfox)"],
        "trigger": ["nik \\(sonicfox\\), sonicfox"],
    },
    "darwen": {"character": ["darwen"], "trigger": ["darwen, digimon"]},
    "edi_(mass_effect)": {
        "character": ["edi_(mass_effect)"],
        "trigger": ["edi \\(mass effect\\), mass effect"],
    },
    "chihiro_ogino": {
        "character": ["chihiro_ogino"],
        "trigger": ["chihiro ogino, ghibli"],
    },
    "jazzotter": {"character": ["jazzotter"], "trigger": ["jazzotter, pokemon"]},
    "kaiman_(dorohedoro)": {
        "character": ["kaiman_(dorohedoro)"],
        "trigger": ["kaiman \\(dorohedoro\\), dorohedoro"],
    },
    "vela_(kagekitsoon)": {
        "character": ["vela_(kagekitsoon)"],
        "trigger": ["vela \\(kagekitsoon\\), patreon"],
    },
    "vel'koz_(lol)": {
        "character": ["vel'koz_(lol)"],
        "trigger": ["vel'koz \\(lol\\), riot games"],
    },
    "mina_(gaturo)": {
        "character": ["mina_(gaturo)"],
        "trigger": ["mina \\(gaturo\\), ghostbusters"],
    },
    "zephyr_(dragon)": {
        "character": ["zephyr_(dragon)"],
        "trigger": ["zephyr \\(dragon\\), mythology"],
    },
    "kazzypoof_(character)": {
        "character": ["kazzypoof_(character)"],
        "trigger": ["kazzypoof \\(character\\), pokemon"],
    },
    "barguest_(tas)": {
        "character": ["barguest_(tas)"],
        "trigger": ["barguest \\(tas\\), lifewonders"],
    },
    "the_red_prince": {
        "character": ["the_red_prince"],
        "trigger": ["the red prince, divinity \\(series\\)"],
    },
    "dark_straw": {
        "character": ["dark_straw"],
        "trigger": ["dark straw, my little pony"],
    },
    "artificer_(risk_of_rain)": {
        "character": ["artificer_(risk_of_rain)"],
        "trigger": ["artificer \\(risk of rain\\), risk of rain"],
    },
    "anne_boonchuy": {
        "character": ["anne_boonchuy"],
        "trigger": ["anne boonchuy, disney"],
    },
    "siege_(arknights)": {
        "character": ["siege_(arknights)"],
        "trigger": ["siege \\(arknights\\), studio montagne"],
    },
    "pink_wolf_(ac_stuart)": {
        "character": ["pink_wolf_(ac_stuart)"],
        "trigger": ["pink wolf \\(ac stuart\\), awoo \\(ac stuart\\)"],
    },
    "miles_(nanoff)": {
        "character": ["miles_(nanoff)"],
        "trigger": ["miles \\(nanoff\\), kiss \\(band\\)"],
    },
    "anbs-02": {"character": ["anbs-02"], "trigger": ["anbs-02, zenonzard"]},
    "adena_(ewgengster)": {
        "character": ["adena_(ewgengster)"],
        "trigger": ["adena \\(ewgengster\\), mythology"],
    },
    "swatch_(deltarune)": {
        "character": ["swatch_(deltarune)"],
        "trigger": ["swatch \\(deltarune\\), undertale \\(series\\)"],
    },
    "rumour_(kitfox-krimson)": {
        "character": ["rumour_(kitfox-krimson)"],
        "trigger": ["rumour \\(kitfox-krimson\\), stolen generation"],
    },
    "jin_qiu_(dislyte)": {
        "character": ["jin_qiu_(dislyte)"],
        "trigger": ["jin qiu \\(dislyte\\), dislyte"],
    },
    "smolder_(lol)": {
        "character": ["smolder_(lol)"],
        "trigger": ["smolder \\(lol\\), riot games"],
    },
    "tess_(jak_and_daxter)": {
        "character": ["tess_(jak_and_daxter)"],
        "trigger": ["tess \\(jak and daxter\\), jak and daxter"],
    },
    "salmy": {"character": ["salmy"], "trigger": ["salmy, mythology"]},
    "aayla_secura": {
        "character": ["aayla_secura"],
        "trigger": ["aayla secura, star wars"],
    },
    "pokey_pierce_(mlp)": {
        "character": ["pokey_pierce_(mlp)"],
        "trigger": ["pokey pierce \\(mlp\\), my little pony"],
    },
    "twillight_(twillightskunk)": {
        "character": ["twillight_(twillightskunk)"],
        "trigger": ["twillight \\(twillightskunk\\), mythology"],
    },
    "blackjack_(fallout_equestria)": {
        "character": ["blackjack_(fallout_equestria)"],
        "trigger": ["blackjack \\(fallout equestria\\), my little pony"],
    },
    "keron": {"character": ["keron"], "trigger": ["keron, pokemon"]},
    "kelly_o'dor": {
        "character": ["kelly_o'dor"],
        "trigger": ["kelly o'dor, zandar's saga"],
    },
    "mitsuhisa_aotsuki": {
        "character": ["mitsuhisa_aotsuki"],
        "trigger": ["mitsuhisa aotsuki, morenatsu"],
    },
    "kaitty": {"character": ["kaitty"], "trigger": ["kaitty, nintendo"]},
    "gage_the_panther": {
        "character": ["gage_the_panther"],
        "trigger": ["gage the panther, mythology"],
    },
    "specimen_8": {
        "character": ["specimen_8"],
        "trigger": ["specimen 8, spooky's jump scare mansion"],
    },
    "narmaya": {"character": ["narmaya"], "trigger": ["narmaya, cygames"]},
    "arianna_altomare": {
        "character": ["arianna_altomare"],
        "trigger": ["arianna altomare, pokemon"],
    },
    "serilde": {"character": ["serilde"], "trigger": ["serilde, mythology"]},
    "renamon_(bacn)": {
        "character": ["renamon_(bacn)"],
        "trigger": ["renamon \\(bacn\\), digimon"],
    },
    "miso_(miso_souperstar)": {
        "character": ["miso_(miso_souperstar)"],
        "trigger": ["miso \\(miso souperstar\\), nintendo"],
    },
    "ricky_landon": {
        "character": ["ricky_landon"],
        "trigger": ["ricky landon, mythology"],
    },
    "brunhilda_(dragalia_lost)": {
        "character": ["brunhilda_(dragalia_lost)"],
        "trigger": ["brunhilda \\(dragalia lost\\), cygames"],
    },
    "nikita_akulov_(nika_sharkeh)": {
        "character": ["nikita_akulov_(nika_sharkeh)"],
        "trigger": ["nikita akulov \\(nika sharkeh\\), source filmmaker"],
    },
    "reina_(sachasketchy)": {
        "character": ["reina_(sachasketchy)"],
        "trigger": ["reina \\(sachasketchy\\), blue sky studios"],
    },
    "sydney_bronson": {
        "character": ["sydney_bronson"],
        "trigger": ["sydney bronson, echo project"],
    },
    "rocco_(tallion)": {
        "character": ["rocco_(tallion)"],
        "trigger": ["rocco \\(tallion\\), jackson guitars"],
    },
    "duncan_(doginacafe)": {
        "character": ["duncan_(doginacafe)"],
        "trigger": ["duncan \\(doginacafe\\), lgbt history month"],
    },
    "salt_(paladins)": {
        "character": ["salt_(paladins)"],
        "trigger": ["salt \\(paladins\\), paladins \\(game\\)"],
    },
    "yuel": {"character": ["yuel"], "trigger": ["yuel, pokemon"]},
    "alice_(alice_in_wonderland)": {
        "character": ["alice_(alice_in_wonderland)"],
        "trigger": ["alice \\(alice in wonderland\\), alice in wonderland"],
    },
    "sherlock_hound": {
        "character": ["sherlock_hound"],
        "trigger": ["sherlock hound, sherlock hound \\(series\\)"],
    },
    "purity_the_hedgehog": {
        "character": ["purity_the_hedgehog"],
        "trigger": ["purity the hedgehog, sonic the hedgehog \\(series\\)"],
    },
    "kagetora_(pop'n_music)": {
        "character": ["kagetora_(pop'n_music)"],
        "trigger": ["kagetora \\(pop'n music\\), pop'n music"],
    },
    "doggieo_(character)": {
        "character": ["doggieo_(character)"],
        "trigger": ["doggieo \\(character\\), father's day"],
    },
    "guildmaster_wigglytuff": {
        "character": ["guildmaster_wigglytuff"],
        "trigger": ["guildmaster wigglytuff, pokemon mystery dungeon"],
    },
    "aleu_moonshadow": {
        "character": ["aleu_moonshadow"],
        "trigger": ["aleu moonshadow, nikorokumitsero"],
    },
    "alexandra_salome": {
        "character": ["alexandra_salome"],
        "trigger": ["alexandra salome, pokemon"],
    },
    "boa_sandersonia": {
        "character": ["boa_sandersonia"],
        "trigger": ["boa sandersonia, one piece"],
    },
    "american_eagle": {
        "character": ["american_eagle"],
        "trigger": ["american eagle, 4th of july"],
    },
    "jack_hyperfreak_(hyperfreak666)": {
        "character": ["jack_hyperfreak_(hyperfreak666)"],
        "trigger": ["jack hyperfreak \\(hyperfreak666\\), my little pony"],
    },
    "basil_(disney)": {
        "character": ["basil_(disney)"],
        "trigger": ["basil \\(disney\\), disney"],
    },
    "nosivi": {"character": ["nosivi"], "trigger": ["nosivi, mythology"]},
    "applejack_(eg)": {
        "character": ["applejack_(eg)"],
        "trigger": ["applejack \\(eg\\), my little pony"],
    },
    "ahuizotl_(mlp)": {
        "character": ["ahuizotl_(mlp)"],
        "trigger": ["ahuizotl \\(mlp\\), my little pony"],
    },
    "exterio": {"character": ["exterio"], "trigger": ["exterio, mythology"]},
    "the_hunter_(bloodborne)": {
        "character": ["the_hunter_(bloodborne)"],
        "trigger": ["the hunter \\(bloodborne\\), sony interactive entertainment"],
    },
    "albedo_(overlord)": {
        "character": ["albedo_(overlord)"],
        "trigger": ["albedo \\(overlord\\), overlord \\(series\\)"],
    },
    "daestra_the_hedgehog": {
        "character": ["daestra_the_hedgehog"],
        "trigger": ["daestra the hedgehog, sonic the hedgehog \\(series\\)"],
    },
    "jack_mckinley": {
        "character": ["jack_mckinley"],
        "trigger": ["jack mckinley, christmas"],
    },
    "tits_(lysergide)": {
        "character": ["tits_(lysergide)"],
        "trigger": ["tits \\(lysergide\\), pokemon"],
    },
    "al_(weaver)": {
        "character": ["al_(weaver)"],
        "trigger": ["al \\(weaver\\), pack street"],
    },
    "deli_(delirost)": {
        "character": ["deli_(delirost)"],
        "trigger": ["deli \\(delirost\\), mythology"],
    },
    "sabari": {"character": ["sabari"], "trigger": ["sabari, pokemon"]},
    "reed_(bearra)": {
        "character": ["reed_(bearra)"],
        "trigger": ["reed \\(bearra\\), christmas"],
    },
    "darnell_(zummeng)": {
        "character": ["darnell_(zummeng)"],
        "trigger": ["darnell \\(zummeng\\), patreon"],
    },
    "riju": {"character": ["riju"], "trigger": ["riju, the legend of zelda"]},
    "loyse": {"character": ["loyse"], "trigger": ["loyse, christmas"]},
    "weiss_(paledrake)": {
        "character": ["weiss_(paledrake)"],
        "trigger": ["weiss \\(paledrake\\), paledrake"],
    },
    "dave_(password)": {
        "character": ["dave_(password)"],
        "trigger": ["dave \\(password\\), password \\(visual novel\\)"],
    },
    "zachariah_(velocitycat)": {
        "character": ["zachariah_(velocitycat)"],
        "trigger": ["zachariah \\(velocitycat\\), animal crossing"],
    },
    "zephyr_the_drake": {
        "character": ["zephyr_the_drake"],
        "trigger": ["zephyr the drake, mythology"],
    },
    "lunlunfox_(character)": {
        "character": ["lunlunfox_(character)"],
        "trigger": ["lunlunfox \\(character\\), mythology"],
    },
    "conductor's_wife_(sonic)": {
        "character": ["conductor's_wife_(sonic)"],
        "trigger": ["conductor's wife \\(sonic\\), sonic the hedgehog \\(series\\)"],
    },
    "ayana": {"character": ["ayana"], "trigger": ["ayana, atung231"]},
    "ninetails_(okami)": {
        "character": ["ninetails_(okami)"],
        "trigger": ["ninetails \\(okami\\), okami \\(capcom\\)"],
    },
    "syandene": {
        "character": ["syandene"],
        "trigger": ["syandene, blender \\(software\\)"],
    },
    "kon_the_knight": {
        "character": ["kon_the_knight"],
        "trigger": ["kon the knight, square enix"],
    },
    "goldie_pheasant": {
        "character": ["goldie_pheasant"],
        "trigger": ["goldie pheasant, don bluth"],
    },
    "koda_(brother_bear)": {
        "character": ["koda_(brother_bear)"],
        "trigger": ["koda \\(brother bear\\), disney"],
    },
    "odie_the_dog": {
        "character": ["odie_the_dog"],
        "trigger": ["odie the dog, garfield \\(series\\)"],
    },
    "experiment_627": {
        "character": ["experiment_627"],
        "trigger": ["experiment 627, disney"],
    },
    "cobalt_(cobaltdawg)": {
        "character": ["cobalt_(cobaltdawg)"],
        "trigger": ["cobalt \\(cobaltdawg\\), disney"],
    },
    "kyouji_(morenatsu)": {
        "character": ["kyouji_(morenatsu)"],
        "trigger": ["kyouji \\(morenatsu\\), morenatsu"],
    },
    "skyla_(pokemon)": {
        "character": ["skyla_(pokemon)"],
        "trigger": ["skyla \\(pokemon\\), pokemon"],
    },
    "bmo": {"character": ["bmo"], "trigger": ["bmo, cartoon network"]},
    "tacoyaki_(character)": {
        "character": ["tacoyaki_(character)"],
        "trigger": ["tacoyaki \\(character\\), tumblr"],
    },
    "modpone": {"character": ["modpone"], "trigger": ["modpone, my little pony"]},
    "shema": {"character": ["shema"], "trigger": ["shema, quest for glory"]},
    "iron_bull": {
        "character": ["iron_bull"],
        "trigger": ["iron bull, electronic arts"],
    },
    "meika_(rimba_racer)": {
        "character": ["meika_(rimba_racer)"],
        "trigger": ["meika \\(rimba racer\\), rimba racer"],
    },
    "milo_(catastrophe)": {
        "character": ["milo_(catastrophe)"],
        "trigger": ["milo \\(catastrophe\\), patreon"],
    },
    "anna_(sailoranna)": {
        "character": ["anna_(sailoranna)"],
        "trigger": ["anna \\(sailoranna\\), mythology"],
    },
    "nakhta": {"character": ["nakhta"], "trigger": ["nakhta, the pirate's fate"]},
    "rosita_(sing)": {
        "character": ["rosita_(sing)"],
        "trigger": ["rosita \\(sing\\), illumination entertainment"],
    },
    "michi_tsuki": {
        "character": ["michi_tsuki"],
        "trigger": ["michi tsuki, square enix"],
    },
    "delta_vee": {"character": ["delta_vee"], "trigger": ["delta vee, my little pony"]},
    "pal_(gym_pals)": {
        "character": ["pal_(gym_pals)"],
        "trigger": ["pal \\(gym pals\\), gym pals"],
    },
    "olli_(braeburned)": {
        "character": ["olli_(braeburned)"],
        "trigger": ["olli \\(braeburned\\), mythology"],
    },
    "tiril": {"character": ["tiril"], "trigger": ["tiril, halloween"]},
    "nekotsuki_kohime": {
        "character": ["nekotsuki_kohime"],
        "trigger": ["nekotsuki kohime, sony interactive entertainment"],
    },
    "crius": {"character": ["crius"], "trigger": ["crius, mythology"]},
    "osamu_tanaka": {"character": ["osamu_tanaka"], "trigger": ["osamu tanaka, ikea"]},
    "gothfield": {
        "character": ["gothfield"],
        "trigger": ["gothfield, garfield \\(series\\)"],
    },
    "bruno_rhynox": {
        "character": ["bruno_rhynox"],
        "trigger": ["bruno rhynox, mythology"],
    },
    "fran_cervice": {
        "character": ["fran_cervice"],
        "trigger": ["fran cervice, christmas"],
    },
    "frank_westerveldt": {
        "character": ["frank_westerveldt"],
        "trigger": ["frank westerveldt, where's waldo?"],
    },
    "stewie_griffin": {
        "character": ["stewie_griffin"],
        "trigger": ["stewie griffin, family guy"],
    },
    "ssvanti": {"character": ["ssvanti"], "trigger": ["ssvanti, everquest"]},
    "frankie_foster": {
        "character": ["frankie_foster"],
        "trigger": ["frankie foster, foster's home for imaginary friends"],
    },
    "mephiles_the_dark": {
        "character": ["mephiles_the_dark"],
        "trigger": ["mephiles the dark, sonic the hedgehog \\(series\\)"],
    },
    "itou_sora": {"character": ["itou_sora"], "trigger": ["itou sora, halloween"]},
    "sheen_(bedfellows)": {
        "character": ["sheen_(bedfellows)"],
        "trigger": ["sheen \\(bedfellows\\), bedfellows"],
    },
    "waluigi": {"character": ["waluigi"], "trigger": ["waluigi, mario bros"]},
    "sheeva": {"character": ["sheeva"], "trigger": ["sheeva, mortal kombat"]},
    "anna_(luke154)": {
        "character": ["anna_(luke154)"],
        "trigger": ["anna \\(luke154\\), warner brothers"],
    },
    "joel_mustard": {
        "character": ["joel_mustard"],
        "trigger": ["joel mustard, patreon"],
    },
    "giga_mermaid": {
        "character": ["giga_mermaid"],
        "trigger": ["giga mermaid, wayforward"],
    },
    "bastian_(leobo)": {
        "character": ["bastian_(leobo)"],
        "trigger": ["bastian \\(leobo\\), patreon"],
    },
    "sunny_way_(character)": {
        "character": ["sunny_way_(character)"],
        "trigger": ["sunny way \\(character\\), mythology"],
    },
    "queen_(alfa995)": {
        "character": ["queen_(alfa995)"],
        "trigger": ["queen \\(alfa995\\), blender \\(software\\)"],
    },
    "mokuji-kun": {
        "character": ["mokuji-kun"],
        "trigger": ["mokuji-kun, little tail bronx"],
    },
    "leto_(letodoesart)": {
        "character": ["leto_(letodoesart)"],
        "trigger": ["leto \\(letodoesart\\), patreon"],
    },
    "sally_hazel": {"character": ["sally_hazel"], "trigger": ["sally hazel, nintendo"]},
    "black_coat_(commissarspuddy)": {
        "character": ["black_coat_(commissarspuddy)"],
        "trigger": ["black coat \\(commissarspuddy\\), mythology"],
    },
    "crayon_(character)": {
        "character": ["crayon_(character)"],
        "trigger": ["crayon \\(character\\), pokemon"],
    },
    "icy_heart": {"character": ["icy_heart"], "trigger": ["icy heart, my little pony"]},
    "dabelette_(character)": {
        "character": ["dabelette_(character)"],
        "trigger": ["dabelette \\(character\\), pokemon"],
    },
    "turo_of_akesh": {
        "character": ["turo_of_akesh"],
        "trigger": ["turo of akesh, mythology"],
    },
    "phobe_(lfswail)": {
        "character": ["phobe_(lfswail)"],
        "trigger": ["phobe \\(lfswail\\), sanrio"],
    },
    "whislash_(arknights)": {
        "character": ["whislash_(arknights)"],
        "trigger": ["whislash \\(arknights\\), studio montagne"],
    },
    "dragonplayer_(character)": {
        "character": ["dragonplayer_(character)"],
        "trigger": ["dragonplayer \\(character\\), blender cycles"],
    },
    "frankenstein's_monster": {
        "character": ["frankenstein's_monster"],
        "trigger": ["frankenstein's monster, halloween"],
    },
    "selina_kyle_(character)": {
        "character": ["selina_kyle_(character)"],
        "trigger": ["selina kyle \\(character\\), dc comics"],
    },
    "shirley_the_loon": {
        "character": ["shirley_the_loon"],
        "trigger": ["shirley the loon, tiny toon adventures"],
    },
    "jeri_katou": {"character": ["jeri_katou"], "trigger": ["jeri katou, digimon"]},
    "rochelle_barnette": {
        "character": ["rochelle_barnette"],
        "trigger": ["rochelle barnette, christmas"],
    },
    "karn_(karn_the_wolf)": {
        "character": ["karn_(karn_the_wolf)"],
        "trigger": ["karn \\(karn the wolf\\), pokemon"],
    },
    "whitney_(pnc)": {
        "character": ["whitney_(pnc)"],
        "trigger": ["whitney \\(pnc\\), peter and company"],
    },
    "amazon_(dragon's_crown)": {
        "character": ["amazon_(dragon's_crown)"],
        "trigger": ["amazon \\(dragon's crown\\), dragon's crown"],
    },
    "professor_starflare_(metal)": {
        "character": ["professor_starflare_(metal)"],
        "trigger": ["professor starflare \\(metal\\), my little pony"],
    },
    "ember_(warframe)": {
        "character": ["ember_(warframe)"],
        "trigger": ["ember \\(warframe\\), warframe"],
    },
    "diesirae": {"character": ["diesirae"], "trigger": ["diesirae, mythology"]},
    "nightmare_(fnaf)": {
        "character": ["nightmare_(fnaf)"],
        "trigger": ["nightmare \\(fnaf\\), scottgames"],
    },
    "kiva_(amazon)": {
        "character": ["kiva_(amazon)"],
        "trigger": ["kiva \\(amazon\\), my little pony"],
    },
    "vapor_trail_(mlp)": {
        "character": ["vapor_trail_(mlp)"],
        "trigger": ["vapor trail \\(mlp\\), my little pony"],
    },
    "zahra_(airheart)": {
        "character": ["zahra_(airheart)"],
        "trigger": ["zahra \\(airheart\\), valentine's day"],
    },
    "cynthia_walker": {
        "character": ["cynthia_walker"],
        "trigger": ["cynthia walker, disney"],
    },
    "sulong_carrot": {
        "character": ["sulong_carrot"],
        "trigger": ["sulong carrot, one piece"],
    },
    "nami_(teranen)": {
        "character": ["nami_(teranen)"],
        "trigger": ["nami \\(teranen\\), mythology"],
    },
    "winter_(wof)": {
        "character": ["winter_(wof)"],
        "trigger": ["winter \\(wof\\), mythology"],
    },
    "piper_(tcitw)": {
        "character": ["piper_(tcitw)"],
        "trigger": ["piper \\(tcitw\\), the cabin in the woods \\(arania\\)"],
    },
    "keith_(tcitw)": {
        "character": ["keith_(tcitw)"],
        "trigger": ["keith \\(tcitw\\), the cabin in the woods \\(arania\\)"],
    },
    "april_(zigzagmag)": {
        "character": ["april_(zigzagmag)"],
        "trigger": ["april \\(zigzagmag\\), mythology"],
    },
    "haruto_arashiki": {
        "character": ["haruto_arashiki"],
        "trigger": ["haruto arashiki, nintendo switch"],
    },
    "racha_(otterjunk)": {
        "character": ["racha_(otterjunk)"],
        "trigger": ["racha \\(otterjunk\\), pokemon"],
    },
    "ryoko_(ryoko-kitsune)": {
        "character": ["ryoko_(ryoko-kitsune)"],
        "trigger": ["ryoko \\(ryoko-kitsune\\), christmas"],
    },
    "cheri_(atrolux)": {
        "character": ["cheri_(atrolux)"],
        "trigger": ["cheri \\(atrolux\\), nintendo"],
    },
    "male_operator": {
        "character": ["male_operator"],
        "trigger": ["male operator, lifewonders"],
    },
    "sins_(sinsquest)": {
        "character": ["sins_(sinsquest)"],
        "trigger": ["sins \\(sinsquest\\), nintendo"],
    },
    "rolo_(rolo_stuff)": {
        "character": ["rolo_(rolo_stuff)"],
        "trigger": ["rolo \\(rolo stuff\\), christmas"],
    },
    "diederich_olsen": {
        "character": ["diederich_olsen"],
        "trigger": ["diederich olsen, knights college"],
    },
    "sebastian_the_husky": {
        "character": ["sebastian_the_husky"],
        "trigger": ["sebastian the husky, mythology"],
    },
    "jerry_(jordo)": {
        "character": ["jerry_(jordo)"],
        "trigger": ["jerry \\(jordo\\), mythology"],
    },
    "shinx_(strikerman)": {
        "character": ["shinx_(strikerman)"],
        "trigger": ["shinx \\(strikerman\\), pokemon"],
    },
    "akuma_gaoru": {
        "character": ["akuma_gaoru"],
        "trigger": ["akuma gaoru, futaba channel"],
    },
    "giran": {"character": ["giran"], "trigger": ["giran, dragon ball"]},
    "ajani_goldmane": {
        "character": ["ajani_goldmane"],
        "trigger": ["ajani goldmane, wizards of the coast"],
    },
    "kemba_kha_regent": {
        "character": ["kemba_kha_regent"],
        "trigger": ["kemba kha regent, wizards of the coast"],
    },
    "mathew_kelly": {
        "character": ["mathew_kelly"],
        "trigger": ["mathew kelly, mythology"],
    },
    "ruska": {"character": ["ruska"], "trigger": ["ruska, family guy"]},
    "bladerush_(character)": {
        "character": ["bladerush_(character)"],
        "trigger": ["bladerush \\(character\\), mythology"],
    },
    "claire_redfield": {
        "character": ["claire_redfield"],
        "trigger": ["claire redfield, resident evil"],
    },
    "killer_queen": {
        "character": ["killer_queen"],
        "trigger": ["killer queen, jojo's bizarre adventure"],
    },
    "therris": {"character": ["therris"], "trigger": ["therris, mestiso"]},
    "joe_kido": {"character": ["joe_kido"], "trigger": ["joe kido, digimon"]},
    "cat_knight": {
        "character": ["cat_knight"],
        "trigger": ["cat knight, omae wa mou shindeiru"],
    },
    "logan_grey": {"character": ["logan_grey"], "trigger": ["logan grey, nintendo"]},
    "officer_mchorn": {
        "character": ["officer_mchorn"],
        "trigger": ["officer mchorn, disney"],
    },
    "lesser_dog": {
        "character": ["lesser_dog"],
        "trigger": ["lesser dog, undertale \\(series\\)"],
    },
    "romeo_(leobo)": {
        "character": ["romeo_(leobo)"],
        "trigger": ["romeo \\(leobo\\), patreon"],
    },
    "corrin": {"character": ["corrin"], "trigger": ["corrin, nintendo"]},
    "haley_sturmbringer_(character)": {
        "character": ["haley_sturmbringer_(character)"],
        "trigger": ["haley sturmbringer \\(character\\), mythology"],
    },
    "captain_celaeno_(mlp)": {
        "character": ["captain_celaeno_(mlp)"],
        "trigger": ["captain celaeno \\(mlp\\), my little pony"],
    },
    "clairen_(rivals_of_aether)": {
        "character": ["clairen_(rivals_of_aether)"],
        "trigger": ["clairen \\(rivals of aether\\), rivals of aether"],
    },
    "chase_(mlp)": {
        "character": ["chase_(mlp)"],
        "trigger": ["chase \\(mlp\\), my little pony"],
    },
    "bailey_(naughtymorg)": {
        "character": ["bailey_(naughtymorg)"],
        "trigger": ["bailey \\(naughtymorg\\), mythology"],
    },
    "leo_(zourik)": {
        "character": ["leo_(zourik)"],
        "trigger": ["leo \\(zourik\\), pokemon"],
    },
    "ram_(reptilligator)": {
        "character": ["ram_(reptilligator)"],
        "trigger": ["ram \\(reptilligator\\), mythology"],
    },
    "obsidian_(lotusgoatess)": {
        "character": ["obsidian_(lotusgoatess)"],
        "trigger": ["obsidian \\(lotusgoatess\\), mythology"],
    },
    "gaz_membrane": {
        "character": ["gaz_membrane"],
        "trigger": ["gaz membrane, invader zim"],
    },
    "sekhmet": {"character": ["sekhmet"], "trigger": ["sekhmet, egyptian mythology"]},
    "poison_ivy": {"character": ["poison_ivy"], "trigger": ["poison ivy, dc comics"]},
    "wolf_nanaki": {
        "character": ["wolf_nanaki"],
        "trigger": ["wolf nanaki, my little pony"],
    },
    "mr._resetti": {
        "character": ["mr._resetti"],
        "trigger": ["mr. resetti, animal crossing"],
    },
    "debbie_dune": {"character": ["debbie_dune"], "trigger": ["debbie dune, disney"]},
    "katarina_du_couteau_(lol)": {
        "character": ["katarina_du_couteau_(lol)"],
        "trigger": ["katarina du couteau \\(lol\\), riot games"],
    },
    "rutherford_(tkongingi)": {
        "character": ["rutherford_(tkongingi)"],
        "trigger": ["rutherford \\(tkongingi\\), mythology"],
    },
    "scarlet_(armello)": {
        "character": ["scarlet_(armello)"],
        "trigger": ["scarlet \\(armello\\), armello"],
    },
    "fanfan": {"character": ["fanfan"], "trigger": ["fanfan, fundoshi's day"]},
    "maypul": {"character": ["maypul"], "trigger": ["maypul, rivals of aether"]},
    "xion_archaeus": {
        "character": ["xion_archaeus"],
        "trigger": ["xion archaeus, sonic the hedgehog \\(series\\)"],
    },
    "elyssia_(armello)": {
        "character": ["elyssia_(armello)"],
        "trigger": ["elyssia \\(armello\\), armello"],
    },
    "jolt_(fuze)": {
        "character": ["jolt_(fuze)"],
        "trigger": ["jolt \\(fuze\\), pokemon"],
    },
    "michelle_(hladilnik)": {
        "character": ["michelle_(hladilnik)"],
        "trigger": ["michelle \\(hladilnik\\), christmas"],
    },
    "ian_dela_cruz": {
        "character": ["ian_dela_cruz"],
        "trigger": ["ian dela cruz, texnatsu"],
    },
    "mama_maria": {"character": ["mama_maria"], "trigger": ["mama maria, christmas"]},
    "zarro_(zarro_the_raichu)": {
        "character": ["zarro_(zarro_the_raichu)"],
        "trigger": ["zarro \\(zarro the raichu\\), pokemon"],
    },
    "auria_jansson": {
        "character": ["auria_jansson"],
        "trigger": ["auria jansson, pokemon"],
    },
    "rhoda_(the_dogsmith)": {
        "character": ["rhoda_(the_dogsmith)"],
        "trigger": ["rhoda \\(the dogsmith\\), christmas"],
    },
    "knavie_(k_navie)": {
        "character": ["knavie_(k_navie)"],
        "trigger": ["knavie \\(k navie\\), mythology"],
    },
    "jin_yorushika": {
        "character": ["jin_yorushika"],
        "trigger": ["jin yorushika, christmas"],
    },
    "sky_(youwannaslap)": {
        "character": ["sky_(youwannaslap)"],
        "trigger": ["sky \\(youwannaslap\\), mythology"],
    },
    "kashino_(azur_lane)": {
        "character": ["kashino_(azur_lane)"],
        "trigger": ["kashino \\(azur lane\\), azur lane"],
    },
    "albafox": {"character": ["albafox"], "trigger": ["albafox, pokemon"]},
    "george_(george701)": {
        "character": ["george_(george701)"],
        "trigger": ["george \\(george701\\), happy tree friends"],
    },
    "luca_paguro": {"character": ["luca_paguro"], "trigger": ["luca paguro, disney"]},
    "aurora_(spacecamper)": {
        "character": ["aurora_(spacecamper)"],
        "trigger": ["aurora \\(spacecamper\\), nintendo"],
    },
    "sody_pop_(chikn_nuggit)": {
        "character": ["sody_pop_(chikn_nuggit)"],
        "trigger": ["sody pop \\(chikn nuggit\\), chikn nuggit"],
    },
    "samantha_(cupcakecarly)": {
        "character": ["samantha_(cupcakecarly)"],
        "trigger": ["samantha \\(cupcakecarly\\), mythology"],
    },
    "edna_(school_days)": {
        "character": ["edna_(school_days)"],
        "trigger": ["edna \\(school days\\), school days"],
    },
    "darth_vader": {
        "character": ["darth_vader"],
        "trigger": ["darth vader, star wars"],
    },
    "donkey_(shrek)": {
        "character": ["donkey_(shrek)"],
        "trigger": ["donkey \\(shrek\\), shrek \\(series\\)"],
    },
    "king_koopa": {"character": ["king_koopa"], "trigger": ["king koopa, mario bros"]},
    "stranger_(mamoru-kun)": {
        "character": ["stranger_(mamoru-kun)"],
        "trigger": ["stranger \\(mamoru-kun\\), little tail bronx"],
    },
    "ren_höek": {"character": ["ren_höek"], "trigger": ["ren höek, ren and stimpy"]},
    "giovanni_(pokemon)": {
        "character": ["giovanni_(pokemon)"],
        "trigger": ["giovanni \\(pokemon\\), team rocket"],
    },
    "roll_(mega_man)": {
        "character": ["roll_(mega_man)"],
        "trigger": ["roll \\(mega man\\), capcom"],
    },
    "meowth_(team_rocket)": {
        "character": ["meowth_(team_rocket)"],
        "trigger": ["meowth \\(team rocket\\), team rocket"],
    },
    "smile.dog": {"character": ["smile.dog"], "trigger": ["smile.dog, creepypasta"]},
    "dnk": {"character": ["dnk"], "trigger": ["dnk, mythology"]},
    "alice_margatroid": {
        "character": ["alice_margatroid"],
        "trigger": ["alice margatroid, touhou"],
    },
    "maxwell_(housepets!)": {
        "character": ["maxwell_(housepets!)"],
        "trigger": ["maxwell \\(housepets!\\), housepets!"],
    },
    "katie_kirster": {
        "character": ["katie_kirster"],
        "trigger": ["katie kirster, pokemon"],
    },
    "fox_(housepets!)": {
        "character": ["fox_(housepets!)"],
        "trigger": ["fox \\(housepets!\\), housepets!"],
    },
    "ludmilla_(bartok)": {
        "character": ["ludmilla_(bartok)"],
        "trigger": ["ludmilla \\(bartok\\), bartok the magnificent"],
    },
    "reginald_(nedroid)": {
        "character": ["reginald_(nedroid)"],
        "trigger": ["reginald \\(nedroid\\), christmas"],
    },
    "fauna_(animal_crossing)": {
        "character": ["fauna_(animal_crossing)"],
        "trigger": ["fauna \\(animal crossing\\), animal crossing"],
    },
    "foxeh": {"character": ["foxeh"], "trigger": ["foxeh, twitter"]},
    "fire_eclipse": {
        "character": ["fire_eclipse"],
        "trigger": ["fire eclipse, my little pony"],
    },
    "arcane_shade": {
        "character": ["arcane_shade"],
        "trigger": ["arcane shade, my little pony"],
    },
    "aura_spark": {"character": ["aura_spark"], "trigger": ["aura spark, mythology"]},
    "thibby_(thibbycat)": {
        "character": ["thibby_(thibbycat)"],
        "trigger": ["thibby \\(thibbycat\\), nintendo"],
    },
    "kaylin": {
        "character": ["kaylin"],
        "trigger": ["kaylin, a million different colors"],
    },
    "maru_(marujawselyn)": {
        "character": ["maru_(marujawselyn)"],
        "trigger": ["maru \\(marujawselyn\\), mythology"],
    },
    "sky_(umbry_sky)": {
        "character": ["sky_(umbry_sky)"],
        "trigger": ["sky \\(umbry sky\\), pokemon"],
    },
    "night_glider_(mlp)": {
        "character": ["night_glider_(mlp)"],
        "trigger": ["night glider \\(mlp\\), my little pony"],
    },
    "scaramouche_rotbelly": {
        "character": ["scaramouche_rotbelly"],
        "trigger": ["scaramouche rotbelly, mythology"],
    },
    "tahoe_(stormwx_wolf)": {
        "character": ["tahoe_(stormwx_wolf)"],
        "trigger": ["tahoe \\(stormwx wolf\\), mythology"],
    },
    "roketchu": {"character": ["roketchu"], "trigger": ["roketchu, mythology"]},
    "lisbelle_(doneru)": {
        "character": ["lisbelle_(doneru)"],
        "trigger": ["lisbelle \\(doneru\\), mythology"],
    },
    "pixxy_fizzleclank": {
        "character": ["pixxy_fizzleclank"],
        "trigger": ["pixxy fizzleclank, warcraft"],
    },
    "zoey_(jwinkz)": {
        "character": ["zoey_(jwinkz)"],
        "trigger": ["zoey \\(jwinkz\\), nintendo"],
    },
    "julia_(vestina)": {
        "character": ["julia_(vestina)"],
        "trigger": ["julia \\(vestina\\), the feast of nero"],
    },
    "helena_(bonk6)": {
        "character": ["helena_(bonk6)"],
        "trigger": ["helena \\(bonk6\\), disney"],
    },
    "tania_marovitch": {
        "character": ["tania_marovitch"],
        "trigger": ["tania marovitch, mythology"],
    },
    "websly": {"character": ["websly"], "trigger": ["websly, nintendo"]},
    "dave_(confusedraven)": {
        "character": ["dave_(confusedraven)"],
        "trigger": ["dave \\(confusedraven\\), piper perri surrounded"],
    },
    "william_(falcon_mccooper)": {
        "character": ["william_(falcon_mccooper)"],
        "trigger": ["william \\(falcon mccooper\\), patreon"],
    },
    "kazusa": {"character": ["kazusa"], "trigger": ["kazusa, tamacolle"]},
    "scorbunny_(valorlynz)": {
        "character": ["scorbunny_(valorlynz)"],
        "trigger": ["scorbunny \\(valorlynz\\), pokemon"],
    },
    "papa_titan_(the_owl_house)": {
        "character": ["papa_titan_(the_owl_house)"],
        "trigger": ["papa titan \\(the owl house\\), disney"],
    },
    "marisa_kirisame": {
        "character": ["marisa_kirisame"],
        "trigger": ["marisa kirisame, touhou"],
    },
    "yukari_yakumo": {
        "character": ["yukari_yakumo"],
        "trigger": ["yukari yakumo, touhou"],
    },
    "furrball": {"character": ["furrball"], "trigger": ["furrball, warner brothers"]},
    "dr._jacques_von_hamsterviel": {
        "character": ["dr._jacques_von_hamsterviel"],
        "trigger": ["dr. jacques von hamsterviel, disney"],
    },
    "ruby_(rubyluvcow)": {
        "character": ["ruby_(rubyluvcow)"],
        "trigger": ["ruby \\(rubyluvcow\\), kingofkof"],
    },
    "little_mac": {"character": ["little_mac"], "trigger": ["little mac, punch-out!!"]},
    "patty_(maple_town)": {
        "character": ["patty_(maple_town)"],
        "trigger": ["patty \\(maple town\\), maple town"],
    },
    "dukey": {"character": ["dukey"], "trigger": ["dukey, johnny test \\(series\\)"]},
    "sabrina_(pokemon)": {
        "character": ["sabrina_(pokemon)"],
        "trigger": ["sabrina \\(pokemon\\), pokemon"],
    },
    "ebonhorn_(foxxeh)": {
        "character": ["ebonhorn_(foxxeh)"],
        "trigger": ["ebonhorn \\(foxxeh\\), mythology"],
    },
    "gali": {"character": ["gali"], "trigger": ["gali, bionicle"]},
    "mr._kat": {"character": ["mr._kat"], "trigger": ["mr. kat, kid vs. kat"]},
    "shirley_the_medium": {
        "character": ["shirley_the_medium"],
        "trigger": ["shirley the medium, cartoon network"],
    },
    "little_strongheart_(mlp)": {
        "character": ["little_strongheart_(mlp)"],
        "trigger": ["little strongheart \\(mlp\\), my little pony"],
    },
    "chief_(animal_crossing)": {
        "character": ["chief_(animal_crossing)"],
        "trigger": ["chief \\(animal crossing\\), animal crossing"],
    },
    "poppy_opossum_(character)": {
        "character": ["poppy_opossum_(character)"],
        "trigger": ["poppy opossum \\(character\\), poppy opossum"],
    },
    "digi": {"character": ["digi"], "trigger": ["digi, mythology"]},
    "crew_(anti_dev)": {
        "character": ["crew_(anti_dev)"],
        "trigger": ["crew \\(anti dev\\), mythology"],
    },
    "morty_smith": {
        "character": ["morty_smith"],
        "trigger": ["morty smith, rick and morty"],
    },
    "zorondo_ron": {
        "character": ["zorondo_ron"],
        "trigger": ["zorondo ron, kaiketsu zorori"],
    },
    "austin_(night_physics)": {
        "character": ["austin_(night_physics)"],
        "trigger": ["austin \\(night physics\\), mythology"],
    },
    "tacet_the_terror": {
        "character": ["tacet_the_terror"],
        "trigger": ["tacet the terror, mythology"],
    },
    "nuwa_nightstone": {
        "character": ["nuwa_nightstone"],
        "trigger": ["nuwa nightstone, mythology"],
    },
    "arvie_dreadmaw": {
        "character": ["arvie_dreadmaw"],
        "trigger": ["arvie dreadmaw, wizards of the coast"],
    },
    "go1den_(wanda_fan_one_piece)": {
        "character": ["go1den_(wanda_fan_one_piece)"],
        "trigger": ["go1den \\(wanda fan one piece\\), one piece"],
    },
    "stephen_wintre": {
        "character": ["stephen_wintre"],
        "trigger": ["stephen wintre, mythology"],
    },
    "cryozen": {"character": ["cryozen"], "trigger": ["cryozen, mythology"]},
    "fatima_eaglefeather": {
        "character": ["fatima_eaglefeather"],
        "trigger": ["fatima eaglefeather, sdorica"],
    },
    "hedgehog_(sci)": {
        "character": ["hedgehog_(sci)"],
        "trigger": ["hedgehog \\(sci\\), cartoon network"],
    },
    "howlitebadger": {
        "character": ["howlitebadger"],
        "trigger": ["howlitebadger, nintendo"],
    },
    "adelgund": {
        "character": ["adelgund"],
        "trigger": ["adelgund, furryfight chronicles"],
    },
    "konigpanther": {
        "character": ["konigpanther"],
        "trigger": ["konigpanther, mythology"],
    },
    "zeeb_wolfy": {"character": ["zeeb_wolfy"], "trigger": ["zeeb wolfy, sextember"]},
    "akumu_(nightmareroa)": {
        "character": ["akumu_(nightmareroa)"],
        "trigger": ["akumu \\(nightmareroa\\), pokemon"],
    },
    "kayla_(lonnyk)": {
        "character": ["kayla_(lonnyk)"],
        "trigger": ["kayla \\(lonnyk\\), snapchat"],
    },
    "rajah_(disney)": {
        "character": ["rajah_(disney)"],
        "trigger": ["rajah \\(disney\\), disney"],
    },
    "eeyore": {
        "character": ["eeyore"],
        "trigger": ["eeyore, winnie the pooh \\(franchise\\)"],
    },
    "myrilla": {"character": ["myrilla"], "trigger": ["myrilla, mythology"]},
    "the_brain": {
        "character": ["the_brain"],
        "trigger": ["the brain, warner brothers"],
    },
    "wardy": {"character": ["wardy"], "trigger": ["wardy, ori \\(series\\)"]},
    "quillu_(character)": {
        "character": ["quillu_(character)"],
        "trigger": ["quillu \\(character\\), mythology"],
    },
    "froggy_(sonic)": {
        "character": ["froggy_(sonic)"],
        "trigger": ["froggy \\(sonic\\), sonic the hedgehog \\(series\\)"],
    },
    "roo_(winnie_the_pooh)": {
        "character": ["roo_(winnie_the_pooh)"],
        "trigger": ["roo \\(winnie the pooh\\), winnie the pooh \\(franchise\\)"],
    },
    "mitsuko_(spacescape)": {
        "character": ["mitsuko_(spacescape)"],
        "trigger": ["mitsuko \\(spacescape\\), spacescape \\(game\\)"],
    },
    "penny_flynn": {
        "character": ["penny_flynn"],
        "trigger": ["penny flynn, hollandworks"],
    },
    "chilly_pepper": {
        "character": ["chilly_pepper"],
        "trigger": ["chilly pepper, my little pony"],
    },
    "rileymutt": {
        "character": ["rileymutt"],
        "trigger": ["rileymutt, superman \\(series\\)"],
    },
    "alice_(floraverse)": {
        "character": ["alice_(floraverse)"],
        "trigger": ["alice \\(floraverse\\), floraverse"],
    },
    "axis_(character)": {
        "character": ["axis_(character)"],
        "trigger": ["axis \\(character\\), mythology"],
    },
    "gakusha": {
        "character": ["gakusha"],
        "trigger": ["gakusha, gamba no bouken \\(series\\)"],
    },
    "xuan_(xuan_sirius)": {
        "character": ["xuan_(xuan_sirius)"],
        "trigger": ["xuan \\(xuan sirius\\), pokemon"],
    },
    "murasaki_(lightsource)": {
        "character": ["murasaki_(lightsource)"],
        "trigger": ["murasaki \\(lightsource\\), egyptian mythology"],
    },
    "diana_(thecon)": {
        "character": ["diana_(thecon)"],
        "trigger": ["diana \\(thecon\\), 4th of july"],
    },
    "toriel_(underfell)": {
        "character": ["toriel_(underfell)"],
        "trigger": ["toriel \\(underfell\\), undertale \\(series\\)"],
    },
    "lydia_hudson": {
        "character": ["lydia_hudson"],
        "trigger": ["lydia hudson, halloween"],
    },
    "susy_sallister": {
        "character": ["susy_sallister"],
        "trigger": ["susy sallister, digimon"],
    },
    "zohara_(reddragonsyndicate)": {
        "character": ["zohara_(reddragonsyndicate)"],
        "trigger": ["zohara \\(reddragonsyndicate\\), bunny and fox world"],
    },
    "marsha_twilight": {
        "character": ["marsha_twilight"],
        "trigger": ["marsha twilight, mythology"],
    },
    "winterwolfy": {"character": ["winterwolfy"], "trigger": ["winterwolfy, pokemon"]},
    "femclaw": {"character": ["femclaw"], "trigger": ["femclaw, fallout"]},
    "jewene_the_ewe": {
        "character": ["jewene_the_ewe"],
        "trigger": ["jewene the ewe, mythology"],
    },
    "scarlett_(whiterabbit95)": {
        "character": ["scarlett_(whiterabbit95)"],
        "trigger": ["scarlett \\(whiterabbit95\\), meme clothing"],
    },
    "jill_(alfa995)": {
        "character": ["jill_(alfa995)"],
        "trigger": ["jill \\(alfa995\\), patreon"],
    },
    "ren_(daikitei)": {
        "character": ["ren_(daikitei)"],
        "trigger": ["ren \\(daikitei\\), nintendo"],
    },
    "sylvane": {"character": ["sylvane"], "trigger": ["sylvane, green lantern"]},
    "gretchen_(kazeattor)": {
        "character": ["gretchen_(kazeattor)"],
        "trigger": ["gretchen \\(kazeattor\\), mythology"],
    },
    "kaiyonato": {"character": ["kaiyonato"], "trigger": ["kaiyonato, mythology"]},
    "rylee_(senimasan)": {
        "character": ["rylee_(senimasan)"],
        "trigger": ["rylee \\(senimasan\\), starbucks"],
    },
    "thylus": {"character": ["thylus"], "trigger": ["thylus, mythology"]},
    "edugta_(character)": {
        "character": ["edugta_(character)"],
        "trigger": ["edugta \\(character\\), mythology"],
    },
    "henry_harris_baxter": {
        "character": ["henry_harris_baxter"],
        "trigger": ["henry harris baxter, mythology"],
    },
    "haan_(character)": {
        "character": ["haan_(character)"],
        "trigger": ["haan \\(character\\), my little pony"],
    },
    "gerry_(dongitos)": {
        "character": ["gerry_(dongitos)"],
        "trigger": ["gerry \\(dongitos\\), nintendo"],
    },
    "liz_(eye_moisturizer)": {
        "character": ["liz_(eye_moisturizer)"],
        "trigger": ["liz \\(eye moisturizer\\), pokemon"],
    },
    "jesse_(falcon_mccooper)": {
        "character": ["jesse_(falcon_mccooper)"],
        "trigger": ["jesse \\(falcon mccooper\\), patreon"],
    },
    "page_(jay-r)": {
        "character": ["page_(jay-r)"],
        "trigger": ["page \\(jay-r\\), nintendo"],
    },
    "sunagawa_(bonedra)": {
        "character": ["sunagawa_(bonedra)"],
        "trigger": ["sunagawa \\(bonedra\\), ikea"],
    },
    "iris_(ratcha)": {
        "character": ["iris_(ratcha)"],
        "trigger": ["iris \\(ratcha\\), mario bros"],
    },
    "eliza_(canisfidelis)": {
        "character": ["eliza_(canisfidelis)"],
        "trigger": ["eliza \\(canisfidelis\\), no nut november"],
    },
    "iyo_(tamacolle)": {
        "character": ["iyo_(tamacolle)"],
        "trigger": ["iyo \\(tamacolle\\), tamacolle"],
    },
    "maxine_(domibun)": {
        "character": ["maxine_(domibun)"],
        "trigger": ["maxine \\(domibun\\), source filmmaker"],
    },
    "left_(atomic_heart)": {
        "character": ["left_(atomic_heart)"],
        "trigger": ["left \\(atomic heart\\), atomic heart"],
    },
    "robert_(pancaketiffy)": {
        "character": ["robert_(pancaketiffy)"],
        "trigger": ["robert \\(pancaketiffy\\), spongebob squarepants"],
    },
    "squillias_(pancaketiffy)": {
        "character": ["squillias_(pancaketiffy)"],
        "trigger": ["squillias \\(pancaketiffy\\), spongebob squarepants"],
    },
    "yamato_iouko": {
        "character": ["yamato_iouko"],
        "trigger": ["yamato iouko, yamatoiouko"],
    },
    "matt_ishida": {"character": ["matt_ishida"], "trigger": ["matt ishida, digimon"]},
    "toodles_galore": {
        "character": ["toodles_galore"],
        "trigger": ["toodles galore, metro-goldwyn-mayer"],
    },
    "amadeus_prower": {
        "character": ["amadeus_prower"],
        "trigger": ["amadeus prower, sonic the hedgehog \\(series\\)"],
    },
    "yuugi_hoshiguma": {
        "character": ["yuugi_hoshiguma"],
        "trigger": ["yuugi hoshiguma, touhou"],
    },
    "storm-tiger": {
        "character": ["storm-tiger"],
        "trigger": ["storm-tiger, mythology"],
    },
    "buster_baxter": {
        "character": ["buster_baxter"],
        "trigger": ["buster baxter, arthur \\(series\\)"],
    },
    "twotail_(mlp)": {
        "character": ["twotail_(mlp)"],
        "trigger": ["twotail \\(mlp\\), my little pony"],
    },
    "lucky_(animal_crossing)": {
        "character": ["lucky_(animal_crossing)"],
        "trigger": ["lucky \\(animal crossing\\), animal crossing"],
    },
    "khris_dragon": {
        "character": ["khris_dragon"],
        "trigger": ["khris dragon, mythology"],
    },
    "charlise_(animal_crossing)": {
        "character": ["charlise_(animal_crossing)"],
        "trigger": ["charlise \\(animal crossing\\), animal crossing"],
    },
    "jazzabelle": {"character": ["jazzabelle"], "trigger": ["jazzabelle, pokemon"]},
    "deer_prince": {
        "character": ["deer_prince"],
        "trigger": ["deer prince, halloween"],
    },
    "triken": {"character": ["triken"], "trigger": ["triken, amagi brilliant park"]},
    "gwen_stacy": {"character": ["gwen_stacy"], "trigger": ["gwen stacy, marvel"]},
    "quentin_(zylo24)": {
        "character": ["quentin_(zylo24)"],
        "trigger": ["quentin \\(zylo24\\), disney"],
    },
    "saki_(tloz)": {
        "character": ["saki_(tloz)"],
        "trigger": ["saki \\(tloz\\), the legend of zelda"],
    },
    "twerkey": {"character": ["twerkey"], "trigger": ["twerkey, miitopia"]},
    "ember_the_typhlosion": {
        "character": ["ember_the_typhlosion"],
        "trigger": ["ember the typhlosion, pokemon"],
    },
    "lovebrew_(oc)": {
        "character": ["lovebrew_(oc)"],
        "trigger": ["lovebrew \\(oc\\), my little pony"],
    },
    "blaze_(chocoscorner)": {
        "character": ["blaze_(chocoscorner)"],
        "trigger": ["blaze \\(chocoscorner\\), pokemon"],
    },
    "lucifer_(hazbin_hotel)": {
        "character": ["lucifer_(hazbin_hotel)"],
        "trigger": ["lucifer \\(hazbin hotel\\), hazbin hotel"],
    },
    "karlach": {"character": ["karlach"], "trigger": ["karlach, electronic arts"]},
    "inuyasha_(inuyasha)": {
        "character": ["inuyasha_(inuyasha)"],
        "trigger": ["inuyasha \\(inuyasha\\), inuyasha"],
    },
    "dio_brando": {
        "character": ["dio_brando"],
        "trigger": ["dio brando, jojo's bizarre adventure"],
    },
    "dion_(doneru)": {
        "character": ["dion_(doneru)"],
        "trigger": ["dion \\(doneru\\), mythology"],
    },
    "hera_(hera)": {
        "character": ["hera_(hera)"],
        "trigger": ["hera \\(hera\\), patreon"],
    },
    "lexi_bunny": {
        "character": ["lexi_bunny"],
        "trigger": ["lexi bunny, looney tunes"],
    },
    "classic_tails": {
        "character": ["classic_tails"],
        "trigger": ["classic tails, sonic the hedgehog \\(series\\)"],
    },
    "mare_do_well_(mlp)": {
        "character": ["mare_do_well_(mlp)"],
        "trigger": ["mare do well \\(mlp\\), my little pony"],
    },
    "armor_king": {"character": ["armor_king"], "trigger": ["armor king, tekken"]},
    "vera_(frisky_ferals)": {
        "character": ["vera_(frisky_ferals)"],
        "trigger": ["vera \\(frisky ferals\\), mythology"],
    },
    "onyx_wasson": {
        "character": ["onyx_wasson"],
        "trigger": ["onyx wasson, onyxtanuki"],
    },
    "tigerstar_(warriors)": {
        "character": ["tigerstar_(warriors)"],
        "trigger": ["tigerstar \\(warriors\\), warriors \\(book series\\)"],
    },
    "kiba_kurokage": {
        "character": ["kiba_kurokage"],
        "trigger": ["kiba kurokage, pokemon"],
    },
    "nymlus": {"character": ["nymlus"], "trigger": ["nymlus, floraverse"]},
    "ariyah_(meg)": {
        "character": ["ariyah_(meg)"],
        "trigger": ["ariyah \\(meg\\), ghibli"],
    },
    "anne_kennel": {
        "character": ["anne_kennel"],
        "trigger": ["anne kennel, pups of liberty"],
    },
    "major_friedkin": {
        "character": ["major_friedkin"],
        "trigger": ["major friedkin, disney"],
    },
    "toumak_(character)": {
        "character": ["toumak_(character)"],
        "trigger": ["toumak \\(character\\), mythology"],
    },
    "zephyr_(tyunre)": {
        "character": ["zephyr_(tyunre)"],
        "trigger": ["zephyr \\(tyunre\\), christmas"],
    },
    "casey_ramser": {
        "character": ["casey_ramser"],
        "trigger": ["casey ramser, texnatsu"],
    },
    "wavern": {"character": ["wavern"], "trigger": ["wavern, bakugan"]},
    "zaryusu_shasha": {
        "character": ["zaryusu_shasha"],
        "trigger": ["zaryusu shasha, overlord \\(series\\)"],
    },
    "tida": {"character": ["tida"], "trigger": ["tida, christmas"]},
    "moonshine_(miso_souperstar)": {
        "character": ["moonshine_(miso_souperstar)"],
        "trigger": ["moonshine \\(miso souperstar\\), bethesda softworks"],
    },
    "moblie_(character)": {
        "character": ["moblie_(character)"],
        "trigger": ["moblie \\(character\\), kinktober"],
    },
    "lya_(jarnqk)": {
        "character": ["lya_(jarnqk)"],
        "trigger": ["lya \\(jarnqk\\), mythology"],
    },
    "darastrix_(ihavexboxlive)": {
        "character": ["darastrix_(ihavexboxlive)"],
        "trigger": ["darastrix \\(ihavexboxlive\\), mythology"],
    },
    "cody_(falcon_mccooper)": {
        "character": ["cody_(falcon_mccooper)"],
        "trigger": ["cody \\(falcon mccooper\\), patreon"],
    },
    "alfie_(wonderslug)": {
        "character": ["alfie_(wonderslug)"],
        "trigger": ["alfie \\(wonderslug\\), mythology"],
    },
    "fecto_elfilis": {
        "character": ["fecto_elfilis"],
        "trigger": ["fecto elfilis, kirby \\(series\\)"],
    },
    "mini_(puppy_in_space)": {
        "character": ["mini_(puppy_in_space)"],
        "trigger": ["mini \\(puppy in space\\), monster energy"],
    },
    "right_(atomic_heart)": {
        "character": ["right_(atomic_heart)"],
        "trigger": ["right \\(atomic heart\\), atomic heart"],
    },
    "karu_(nu:_carnival)": {
        "character": ["karu_(nu:_carnival)"],
        "trigger": ["karu \\(nu: carnival\\), nu: carnival"],
    },
    "darke_katt": {
        "character": ["darke_katt"],
        "trigger": ["darke katt, furafterdark"],
    },
    "lion_sora": {"character": ["lion_sora"], "trigger": ["lion sora, kingdom hearts"]},
    "orca_(dc)": {"character": ["orca_(dc)"], "trigger": ["orca \\(dc\\), dc comics"]},
    "aryani": {"character": ["aryani"], "trigger": ["aryani, mythology"]},
    "jazmin_usagi": {
        "character": ["jazmin_usagi"],
        "trigger": ["jazmin usagi, rascals"],
    },
    "shephard": {"character": ["shephard"], "trigger": ["shephard, mythology"]},
    "su_wu": {"character": ["su_wu"], "trigger": ["su wu, kung fu panda"]},
    "sugar_sprinkles": {
        "character": ["sugar_sprinkles"],
        "trigger": ["sugar sprinkles, hasbro"],
    },
    "typhek": {"character": ["typhek"], "trigger": ["typhek, source filmmaker"]},
    "zavok": {
        "character": ["zavok"],
        "trigger": ["zavok, sonic the hedgehog \\(series\\)"],
    },
    "adrian_(crovirus)": {
        "character": ["adrian_(crovirus)"],
        "trigger": ["adrian \\(crovirus\\), mythology"],
    },
    "perci_the_bandicoot": {
        "character": ["perci_the_bandicoot"],
        "trigger": ["perci the bandicoot, sonic the hedgehog \\(series\\)"],
    },
    "softdiamond": {
        "character": ["softdiamond"],
        "trigger": ["softdiamond, mythology"],
    },
    "homura_(homura_kasuka)": {
        "character": ["homura_(homura_kasuka)"],
        "trigger": ["homura \\(homura kasuka\\), nintendo"],
    },
    "linkle": {"character": ["linkle"], "trigger": ["linkle, hyrule warriors"]},
    "grace_mustang": {
        "character": ["grace_mustang"],
        "trigger": ["grace mustang, pokemon"],
    },
    "felimon": {"character": ["felimon"], "trigger": ["felimon, pokemon"]},
    "zeha": {"character": ["zeha"], "trigger": ["zeha, christmas"]},
    "exenthal": {"character": ["exenthal"], "trigger": ["exenthal, mythology"]},
    "adam_caro": {"character": ["adam_caro"], "trigger": ["adam caro, texnatsu"]},
    "amalia_(claralaine)": {
        "character": ["amalia_(claralaine)"],
        "trigger": ["amalia \\(claralaine\\), patreon"],
    },
    "nirimer": {"character": ["nirimer"], "trigger": ["nirimer, mythology"]},
    "shaq_(meatshaq)": {
        "character": ["shaq_(meatshaq)"],
        "trigger": ["shaq \\(meatshaq\\), mythology"],
    },
    "vanessa_(furryrex)": {
        "character": ["vanessa_(furryrex)"],
        "trigger": ["vanessa \\(furryrex\\), warcraft"],
    },
    "dominic_armois": {
        "character": ["dominic_armois"],
        "trigger": ["dominic armois, pokemon"],
    },
    "cayenne_(kasdaq)": {
        "character": ["cayenne_(kasdaq)"],
        "trigger": ["cayenne \\(kasdaq\\), petruz \\(copyright\\)"],
    },
    "kyle_bavarois": {
        "character": ["kyle_bavarois"],
        "trigger": ["kyle bavarois, fuga: melodies of steel"],
    },
    "idena_(swordfox)": {
        "character": ["idena_(swordfox)"],
        "trigger": ["idena \\(swordfox\\), pokemon"],
    },
    "mewtowo_(shadman)": {
        "character": ["mewtowo_(shadman)"],
        "trigger": ["mewtowo \\(shadman\\), pokemon"],
    },
    "ganachethehorse": {
        "character": ["ganachethehorse"],
        "trigger": ["ganachethehorse, mythology"],
    },
    "orangusnake": {
        "character": ["orangusnake"],
        "trigger": ["orangusnake, cartoon network"],
    },
    "coby_(mao_mao)": {
        "character": ["coby_(mao_mao)"],
        "trigger": ["coby \\(mao mao\\), cartoon network"],
    },
    "polly_plantar": {
        "character": ["polly_plantar"],
        "trigger": ["polly plantar, disney"],
    },
    "megan_ziegler": {
        "character": ["megan_ziegler"],
        "trigger": ["megan ziegler, risk of rain"],
    },
    "aoniya_yuudai": {
        "character": ["aoniya_yuudai"],
        "trigger": ["aoniya yuudai, shampoo challenge"],
    },
    "reneigh_(animal_crossing)": {
        "character": ["reneigh_(animal_crossing)"],
        "trigger": ["reneigh \\(animal crossing\\), animal crossing"],
    },
    "sean-zee_petit": {
        "character": ["sean-zee_petit"],
        "trigger": ["sean-zee petit, mythology"],
    },
    "dmitrei": {"character": ["dmitrei"], "trigger": ["dmitrei, mythology"]},
    "ranok_(far_beyond_the_world)": {
        "character": ["ranok_(far_beyond_the_world)"],
        "trigger": [
            "ranok \\(far beyond the world\\), far beyond the world \\(series\\)"
        ],
    },
    "maria_(pancarta)": {
        "character": ["maria_(pancarta)"],
        "trigger": ["maria \\(pancarta\\), pokemon"],
    },
    "oro_(oro97)": {
        "character": ["oro_(oro97)"],
        "trigger": ["oro \\(oro97\\), mythology"],
    },
    "miko_(abz)": {"character": ["miko_(abz)"], "trigger": ["miko \\(abz\\), abz"]},
    "annabee_(woebeeme)": {
        "character": ["annabee_(woebeeme)"],
        "trigger": ["annabee \\(woebeeme\\), bug fables"],
    },
    "amethyst_(kitfox-crimson)": {
        "character": ["amethyst_(kitfox-crimson)"],
        "trigger": ["amethyst \\(kitfox-crimson\\), in our shadow"],
    },
    "schizo_chan_(snoot_game)": {
        "character": ["schizo_chan_(snoot_game)"],
        "trigger": ["schizo chan \\(snoot game\\), cavemanon studios"],
    },
    "canxue_(character)": {
        "character": ["canxue_(character)"],
        "trigger": ["canxue \\(character\\), ubisoft"],
    },
    "segremores": {"character": ["segremores"], "trigger": ["segremores, mythology"]},
    "barby_koala": {
        "character": ["barby_koala"],
        "trigger": ["barby koala, sonic the hedgehog \\(series\\)"],
    },
    "thrall_(warcraft)": {
        "character": ["thrall_(warcraft)"],
        "trigger": ["thrall \\(warcraft\\), warcraft"],
    },
    "queen_aleena_hedgehog": {
        "character": ["queen_aleena_hedgehog"],
        "trigger": ["queen aleena hedgehog, sonic underground"],
    },
    "buster_(lady_and_the_tramp)": {
        "character": ["buster_(lady_and_the_tramp)"],
        "trigger": ["buster \\(lady and the tramp\\), disney"],
    },
    "planeswalker": {
        "character": ["planeswalker"],
        "trigger": ["planeswalker, magic: the gathering"],
    },
    "starswirl_the_bearded_(mlp)": {
        "character": ["starswirl_the_bearded_(mlp)"],
        "trigger": ["starswirl the bearded \\(mlp\\), my little pony"],
    },
    "ambrosia": {"character": ["ambrosia"], "trigger": ["ambrosia, disney"]},
    "bieesha": {"character": ["bieesha"], "trigger": ["bieesha, patreon"]},
    "argent": {"character": ["argent"], "trigger": ["argent, mythology"]},
    "itsuki_(hane)": {
        "character": ["itsuki_(hane)"],
        "trigger": ["itsuki \\(hane\\), handymonsters"],
    },
    "lucina": {"character": ["lucina"], "trigger": ["lucina, nintendo"]},
    "adrian_(firewolf)": {
        "character": ["adrian_(firewolf)"],
        "trigger": ["adrian \\(firewolf\\), alpha knows best"],
    },
    "jess_(kinaj)": {
        "character": ["jess_(kinaj)"],
        "trigger": ["jess \\(kinaj\\), christmas"],
    },
    "marco_diaz": {"character": ["marco_diaz"], "trigger": ["marco diaz, disney"]},
    "fuku_fire": {
        "character": ["fuku_fire"],
        "trigger": ["fuku fire, undertale \\(series\\)"],
    },
    "chelsi": {"character": ["chelsi"], "trigger": ["chelsi, nintendo"]},
    "abby_doug": {"character": ["abby_doug"], "trigger": ["abby doug, h.w.t. studios"]},
    "inkh": {"character": ["inkh"], "trigger": ["inkh, mythology"]},
    "cosma_(ok_k.o.!_lbh)": {
        "character": ["cosma_(ok_k.o.!_lbh)"],
        "trigger": ["cosma \\(ok k.o.! lbh\\), cartoon network"],
    },
    "ms._renee_l'noire": {
        "character": ["ms._renee_l'noire"],
        "trigger": ["ms. renee l'noire, the scream"],
    },
    "ken_(claralaine)": {
        "character": ["ken_(claralaine)"],
        "trigger": ["ken \\(claralaine\\), patreon"],
    },
    "frostbite_(rubberbuns)": {
        "character": ["frostbite_(rubberbuns)"],
        "trigger": ["frostbite \\(rubberbuns\\), christmas"],
    },
    "dakota_(kaggy1)": {
        "character": ["dakota_(kaggy1)"],
        "trigger": ["dakota \\(kaggy1\\), paledrake"],
    },
    "cobra_(petruz)": {
        "character": ["cobra_(petruz)"],
        "trigger": ["cobra \\(petruz\\), petruz \\(copyright\\)"],
    },
    "wolf_(parasitedeath)": {
        "character": ["wolf_(parasitedeath)"],
        "trigger": ["wolf \\(parasitedeath\\), nintendo"],
    },
    "vox_(hazbin_hotel)": {
        "character": ["vox_(hazbin_hotel)"],
        "trigger": ["vox \\(hazbin hotel\\), hazbin hotel"],
    },
    "rotom_phone": {"character": ["rotom_phone"], "trigger": ["rotom phone, pokemon"]},
    "aode_(asonix)": {
        "character": ["aode_(asonix)"],
        "trigger": ["aode \\(asonix\\), disney"],
    },
    "nick_(beez)": {
        "character": ["nick_(beez)"],
        "trigger": ["nick \\(beez\\), patreon"],
    },
    "nik_(nik159)": {
        "character": ["nik_(nik159)"],
        "trigger": ["nik \\(nik159\\), twitter"],
    },
    "nimbus_(world_flipper)": {
        "character": ["nimbus_(world_flipper)"],
        "trigger": ["nimbus \\(world flipper\\), cygames"],
    },
    "dale_(ponehanon)": {
        "character": ["dale_(ponehanon)"],
        "trigger": ["dale \\(ponehanon\\), big bun burgers"],
    },
    "akashi_(live_a_hero)": {
        "character": ["akashi_(live_a_hero)"],
        "trigger": ["akashi \\(live a hero\\), lifewonders"],
    },
    "klee_(genshin_impact)": {
        "character": ["klee_(genshin_impact)"],
        "trigger": ["klee \\(genshin impact\\), mihoyo"],
    },
    "catoblepas_(tas)": {
        "character": ["catoblepas_(tas)"],
        "trigger": ["catoblepas \\(tas\\), lifewonders"],
    },
    "donut_(misterdonut)": {
        "character": ["donut_(misterdonut)"],
        "trigger": ["donut \\(misterdonut\\), pokemon"],
    },
    "tea_party_style_glaceon": {
        "character": ["tea_party_style_glaceon"],
        "trigger": ["tea party style glaceon, pokemon"],
    },
    "hudson_(gargoyles)": {
        "character": ["hudson_(gargoyles)"],
        "trigger": ["hudson \\(gargoyles\\), disney"],
    },
    "tiara_boobowski": {
        "character": ["tiara_boobowski"],
        "trigger": ["tiara boobowski, sonic the hedgehog \\(series\\)"],
    },
    "sheldon_j._plankton": {
        "character": ["sheldon_j._plankton"],
        "trigger": ["sheldon j. plankton, spongebob squarepants"],
    },
    "maggie_reed_(gargoyles)": {
        "character": ["maggie_reed_(gargoyles)"],
        "trigger": ["maggie reed \\(gargoyles\\), disney"],
    },
    "top_cat": {"character": ["top_cat"], "trigger": ["top cat, top cat \\(series\\)"]},
    "choo-choo_(top_cat)": {
        "character": ["choo-choo_(top_cat)"],
        "trigger": ["choo-choo \\(top cat\\), top cat \\(series\\)"],
    },
    "chichi": {"character": ["chichi"], "trigger": ["chichi, dragon ball"]},
    "rein_(amaterasu1)": {
        "character": ["rein_(amaterasu1)"],
        "trigger": ["rein \\(amaterasu1\\), mythology"],
    },
    "sangie_nativus": {
        "character": ["sangie_nativus"],
        "trigger": ["sangie nativus, little tail bronx"],
    },
    "marcus_(rukis)": {
        "character": ["marcus_(rukis)"],
        "trigger": ["marcus \\(rukis\\), unconditional \\(comic\\)"],
    },
    "renard_queenston": {
        "character": ["renard_queenston"],
        "trigger": ["renard queenston, lapfox trax"],
    },
    "kowalski_(madagascar)": {
        "character": ["kowalski_(madagascar)"],
        "trigger": ["kowalski \\(madagascar\\), dreamworks"],
    },
    "daisy_(mlp)": {
        "character": ["daisy_(mlp)"],
        "trigger": ["daisy \\(mlp\\), my little pony"],
    },
    "syx": {"character": ["syx"], "trigger": ["syx, manaworld"]},
    "sweetie_bot_(mlp)": {
        "character": ["sweetie_bot_(mlp)"],
        "trigger": ["sweetie bot \\(mlp\\), my little pony"],
    },
    "noguchi": {"character": ["noguchi"], "trigger": ["noguchi, tooboe bookmark"]},
    "costom10_(character)": {
        "character": ["costom10_(character)"],
        "trigger": ["costom10 \\(character\\), mythology"],
    },
    "weiss_schnee": {"character": ["weiss_schnee"], "trigger": ["weiss schnee, rwby"]},
    "diana_(bashfulsprite)": {
        "character": ["diana_(bashfulsprite)"],
        "trigger": ["diana \\(bashfulsprite\\), mythology"],
    },
    "oracle_(vhsdaii)": {
        "character": ["oracle_(vhsdaii)"],
        "trigger": ["oracle \\(vhsdaii\\), patreon"],
    },
    "quaise_(doneru)": {
        "character": ["quaise_(doneru)"],
        "trigger": ["quaise \\(doneru\\), mythology"],
    },
    "beluinus": {"character": ["beluinus"], "trigger": ["beluinus, mythology"]},
    "violet_echo": {
        "character": ["violet_echo"],
        "trigger": ["violet echo, mythology"],
    },
    "darth_vader_sanchez_(housepets!)": {
        "character": ["darth_vader_sanchez_(housepets!)"],
        "trigger": ["darth vader sanchez \\(housepets!\\), housepets!"],
    },
    "kc_(kingcrazy)": {
        "character": ["kc_(kingcrazy)"],
        "trigger": ["kc \\(kingcrazy\\), nintendo"],
    },
    "matt_donovan": {"character": ["matt_donovan"], "trigger": ["matt donovan, kda"]},
    "kitt_kitan_(character)": {
        "character": ["kitt_kitan_(character)"],
        "trigger": ["kitt kitan \\(character\\), sonic the hedgehog \\(series\\)"],
    },
    "yami_the_veemon": {
        "character": ["yami_the_veemon"],
        "trigger": ["yami the veemon, digimon"],
    },
    "pheronoa": {"character": ["pheronoa"], "trigger": ["pheronoa, pokemon"]},
    "diego_abel": {"character": ["diego_abel"], "trigger": ["diego abel, texnatsu"]},
    "rivas_(yuguni)": {
        "character": ["rivas_(yuguni)"],
        "trigger": ["rivas \\(yuguni\\), mythology"],
    },
    "shiori_(kurus)": {
        "character": ["shiori_(kurus)"],
        "trigger": ["shiori \\(kurus\\), nintendo"],
    },
    "kaylee_(study_partners)": {
        "character": ["kaylee_(study_partners)"],
        "trigger": ["kaylee \\(study partners\\), study partners"],
    },
    "fi_(blen_bodega)": {
        "character": ["fi_(blen_bodega)"],
        "trigger": ["fi \\(blen bodega\\), golden week"],
    },
    "sunny_flowers": {
        "character": ["sunny_flowers"],
        "trigger": ["sunny flowers, pokemon"],
    },
    "satoshi_nagashima_(odd_taxi)": {
        "character": ["satoshi_nagashima_(odd_taxi)"],
        "trigger": ["satoshi nagashima \\(odd taxi\\), odd taxi"],
    },
    "catnip_(khatnid)": {
        "character": ["catnip_(khatnid)"],
        "trigger": ["catnip \\(khatnid\\), riot games"],
    },
    "zorayas": {"character": ["zorayas"], "trigger": ["zorayas, fromsoftware"]},
    "capri_(deerkid)": {
        "character": ["capri_(deerkid)"],
        "trigger": ["capri \\(deerkid\\), halloween"],
    },
    "rose_(fairyfud)": {
        "character": ["rose_(fairyfud)"],
        "trigger": ["rose \\(fairyfud\\), pokemon"],
    },
    "judee": {"character": ["judee"], "trigger": ["judee, cavemanon studios"]},
    "tsenaya": {"character": ["tsenaya"], "trigger": ["tsenaya, patreon"]},
    "king_(cave_story)": {
        "character": ["king_(cave_story)"],
        "trigger": ["king \\(cave story\\), cave story"],
    },
    "jane_read": {
        "character": ["jane_read"],
        "trigger": ["jane read, arthur \\(series\\)"],
    },
    "infested_kerrigan": {
        "character": ["infested_kerrigan"],
        "trigger": ["infested kerrigan, starcraft"],
    },
    "aurenn": {"character": ["aurenn"], "trigger": ["aurenn, mythology"]},
    "penny_(inspector_gadget)": {
        "character": ["penny_(inspector_gadget)"],
        "trigger": ["penny \\(inspector gadget\\), inspector gadget \\(franchise\\)"],
    },
    "dark_magician_girl": {
        "character": ["dark_magician_girl"],
        "trigger": ["dark magician girl, yu-gi-oh!"],
    },
    "ramona_alvarez": {
        "character": ["ramona_alvarez"],
        "trigger": ["ramona alvarez, cookie clicker"],
    },
    "daisy_dingo": {
        "character": ["daisy_dingo"],
        "trigger": ["daisy dingo, blinky bill \\(series\\)"],
    },
    "mattie_(chimangetsu)": {
        "character": ["mattie_(chimangetsu)"],
        "trigger": ["mattie \\(chimangetsu\\), chimangetsu"],
    },
    "iron_man": {"character": ["iron_man"], "trigger": ["iron man, marvel"]},
    "screwball_(mlp)": {
        "character": ["screwball_(mlp)"],
        "trigger": ["screwball \\(mlp\\), my little pony"],
    },
    "lokkun": {"character": ["lokkun"], "trigger": ["lokkun, meme clothing"]},
    "russell_ferguson": {
        "character": ["russell_ferguson"],
        "trigger": ["russell ferguson, hasbro"],
    },
    "fukami_youhei": {
        "character": ["fukami_youhei"],
        "trigger": ["fukami youhei, mekko rarekko"],
    },
    "knight_(towergirls)": {
        "character": ["knight_(towergirls)"],
        "trigger": ["knight \\(towergirls\\), towergirls"],
    },
    "reyathae": {"character": ["reyathae"], "trigger": ["reyathae, pokemon"]},
    "rose_(skybluefox)": {
        "character": ["rose_(skybluefox)"],
        "trigger": ["rose \\(skybluefox\\), pokemon"],
    },
    "chikiot": {"character": ["chikiot"], "trigger": ["chikiot, mythology"]},
    "zoma": {"character": ["zoma"], "trigger": ["zoma, mythology"]},
    "alpha_garza_(vimhomeless)": {
        "character": ["alpha_garza_(vimhomeless)"],
        "trigger": ["alpha garza \\(vimhomeless\\), super planet dolan"],
    },
    "reina_(hypnofood)": {
        "character": ["reina_(hypnofood)"],
        "trigger": ["reina \\(hypnofood\\), the jungle book"],
    },
    "jodira": {"character": ["jodira"], "trigger": ["jodira, mythology"]},
    "pro_bun_(hladilnik)": {
        "character": ["pro_bun_(hladilnik)"],
        "trigger": ["pro bun \\(hladilnik\\), fallout"],
    },
    "bearphones": {"character": ["bearphones"], "trigger": ["bearphones, nintendo"]},
    "broderick_longshanks": {
        "character": ["broderick_longshanks"],
        "trigger": ["broderick longshanks, mythology"],
    },
    "zempy": {"character": ["zempy"], "trigger": ["zempy, mythology"]},
    "brianne_(spikedmauler)": {
        "character": ["brianne_(spikedmauler)"],
        "trigger": ["brianne \\(spikedmauler\\), mythology"],
    },
    "blossom_(battlerite)": {
        "character": ["blossom_(battlerite)"],
        "trigger": ["blossom \\(battlerite\\), battlerite"],
    },
    "diana_rayablanca": {
        "character": ["diana_rayablanca"],
        "trigger": ["diana rayablanca, disney"],
    },
    "fur_hire": {"character": ["fur_hire"], "trigger": ["fur hire, yu-gi-oh!"]},
    "buttons_(milachu92)": {
        "character": ["buttons_(milachu92)"],
        "trigger": ["buttons \\(milachu92\\), pokemon"],
    },
    "shira_kaisuri": {
        "character": ["shira_kaisuri"],
        "trigger": ["shira kaisuri, pokemon"],
    },
    "beenic": {"character": ["beenic"], "trigger": ["beenic, gyee"]},
    "ethan_(grinn3r)": {
        "character": ["ethan_(grinn3r)"],
        "trigger": ["ethan \\(grinn3r\\), nintendo"],
    },
    "salem_(discordthege)": {
        "character": ["salem_(discordthege)"],
        "trigger": ["salem \\(discordthege\\), my little pony"],
    },
    "rogelio_(she-ra)": {
        "character": ["rogelio_(she-ra)"],
        "trigger": ["rogelio \\(she-ra\\), mattel"],
    },
    "taigaxholic": {"character": ["taigaxholic"], "trigger": ["taigaxholic, vtuber"]},
    "saber_ibuki-douji": {
        "character": ["saber_ibuki-douji"],
        "trigger": ["saber ibuki-douji, fate \\(series\\)"],
    },
    "lilithdog": {"character": ["lilithdog"], "trigger": ["lilithdog, mythology"]},
    "wolf_(we_baby_bears)": {
        "character": ["wolf_(we_baby_bears)"],
        "trigger": ["wolf \\(we baby bears\\), cartoon network"],
    },
    "rinny_(character)": {
        "character": ["rinny_(character)"],
        "trigger": ["rinny \\(character\\), blender \\(software\\)"],
    },
    "inv_(rain_world)": {
        "character": ["inv_(rain_world)"],
        "trigger": ["inv \\(rain world\\), videocult"],
    },
    "caine_(tadc)": {
        "character": ["caine_(tadc)"],
        "trigger": ["caine \\(tadc\\), the amazing digital circus"],
    },
    "moonlight_flower": {
        "character": ["moonlight_flower"],
        "trigger": ["moonlight flower, ragnarok online"],
    },
    "sheik_(tloz)": {
        "character": ["sheik_(tloz)"],
        "trigger": ["sheik \\(tloz\\), the legend of zelda"],
    },
    "foxglove_(cdrr)": {
        "character": ["foxglove_(cdrr)"],
        "trigger": ["foxglove \\(cdrr\\), disney"],
    },
    "buttertoast": {
        "character": ["buttertoast"],
        "trigger": ["buttertoast, build tiger"],
    },
    "lady_rainicorn": {
        "character": ["lady_rainicorn"],
        "trigger": ["lady rainicorn, cartoon network"],
    },
    "sapphire_shores_(mlp)": {
        "character": ["sapphire_shores_(mlp)"],
        "trigger": ["sapphire shores \\(mlp\\), my little pony"],
    },
    "luka_cross": {"character": ["luka_cross"], "trigger": ["luka cross, mythology"]},
    "shirokuma": {"character": ["shirokuma"], "trigger": ["shirokuma, shirokuma cafe"]},
    "garret_(rain-yatsu)": {
        "character": ["garret_(rain-yatsu)"],
        "trigger": ["garret \\(rain-yatsu\\), seattle fur"],
    },
    "jegermaistro": {
        "character": ["jegermaistro"],
        "trigger": ["jegermaistro, mythology"],
    },
    "teddy_conner": {
        "character": ["teddy_conner"],
        "trigger": ["teddy conner, raven wolf"],
    },
    "fukuzawa": {"character": ["fukuzawa"], "trigger": ["fukuzawa, tooboe bookmark"]},
    "trenderhoof_(mlp)": {
        "character": ["trenderhoof_(mlp)"],
        "trigger": ["trenderhoof \\(mlp\\), my little pony"],
    },
    "monstercat": {
        "character": ["monstercat"],
        "trigger": ["monstercat, monstercat media"],
    },
    "akhlys": {"character": ["akhlys"], "trigger": ["akhlys, mythology"]},
    "roger_(mike_sherman)": {
        "character": ["roger_(mike_sherman)"],
        "trigger": ["roger \\(mike sherman\\), osamu tezuka"],
    },
    "eerie_(telemonster)": {
        "character": ["eerie_(telemonster)"],
        "trigger": ["eerie \\(telemonster\\), telemonster"],
    },
    "v-0-1-d": {"character": ["v-0-1-d"], "trigger": ["v-0-1-d, mythology"]},
    "asyr": {"character": ["asyr"], "trigger": ["asyr, mythology"]},
    "vizlet": {"character": ["vizlet"], "trigger": ["vizlet, out-of-placers"]},
    "apogee": {"character": ["apogee"], "trigger": ["apogee, my little pony"]},
    "francis_misztalski": {
        "character": ["francis_misztalski"],
        "trigger": ["francis misztalski, halloween"],
    },
    "blake_jackson": {
        "character": ["blake_jackson"],
        "trigger": ["blake jackson, texnatsu"],
    },
    "naomi_minette": {
        "character": ["naomi_minette"],
        "trigger": ["naomi minette, pokemon"],
    },
    "shin_mao": {"character": ["shin_mao"], "trigger": ["shin mao, cartoon network"]},
    "liam_(fuf)": {"character": ["liam_(fuf)"], "trigger": ["liam \\(fuf\\), pokemon"]},
    "red_(topazknight)": {
        "character": ["red_(topazknight)"],
        "trigger": ["red \\(topazknight\\), minecraft"],
    },
    "hanako_(lyorenth-the-dragon)": {
        "character": ["hanako_(lyorenth-the-dragon)"],
        "trigger": ["hanako \\(lyorenth-the-dragon\\), pokemon"],
    },
    "san_inukai": {"character": ["san_inukai"], "trigger": ["san inukai, christmas"]},
    "torrent_(elden_ring)": {
        "character": ["torrent_(elden_ring)"],
        "trigger": ["torrent \\(elden ring\\), fromsoftware"],
    },
    "ruger": {"character": ["ruger"], "trigger": ["ruger, clubstripes"]},
    "valerie": {"character": ["valerie"], "trigger": ["valerie, crowjob in space"]},
    "lucas_(pokemon)": {
        "character": ["lucas_(pokemon)"],
        "trigger": ["lucas \\(pokemon\\), pokemon"],
    },
    "mazoga_the_orc": {
        "character": ["mazoga_the_orc"],
        "trigger": ["mazoga the orc, bethesda softworks"],
    },
    "dracula": {"character": ["dracula"], "trigger": ["dracula, konami"]},
    "forsaken_(character)": {
        "character": ["forsaken_(character)"],
        "trigger": ["forsaken \\(character\\), my little pony"],
    },
    "ms._mowz": {"character": ["ms._mowz"], "trigger": ["ms. mowz, mario bros"]},
    "sligar": {"character": ["sligar"], "trigger": ["sligar, alxias"]},
    "darth_talon": {
        "character": ["darth_talon"],
        "trigger": ["darth talon, star wars"],
    },
    "nova_(meganovav1)": {
        "character": ["nova_(meganovav1)"],
        "trigger": ["nova \\(meganovav1\\), pokemon"],
    },
    "meer": {"character": ["meer"], "trigger": ["meer, mythology"]},
    "weekly": {"character": ["weekly"], "trigger": ["weekly, blacksad"]},
    "forebucks": {"character": ["forebucks"], "trigger": ["forebucks, forepawz"]},
    "thane_(armello)": {
        "character": ["thane_(armello)"],
        "trigger": ["thane \\(armello\\), armello"],
    },
    "daniel_segja": {
        "character": ["daniel_segja"],
        "trigger": ["daniel segja, patreon"],
    },
    "syrth": {"character": ["syrth"], "trigger": ["syrth, east asian mythology"]},
    "cheshire_thaddeus_felonious": {
        "character": ["cheshire_thaddeus_felonious"],
        "trigger": ["cheshire thaddeus felonious, sonic the hedgehog \\(series\\)"],
    },
    "petresko_(character)": {
        "character": ["petresko_(character)"],
        "trigger": ["petresko \\(character\\), mass effect"],
    },
    "vanth": {"character": ["vanth"], "trigger": ["vanth, dreamkeepers"]},
    "ralek_(oc)": {
        "character": ["ralek_(oc)"],
        "trigger": ["ralek \\(oc\\), mythology"],
    },
    "fabienne_growley": {
        "character": ["fabienne_growley"],
        "trigger": ["fabienne growley, disney"],
    },
    "red_guy_(dhmis)": {
        "character": ["red_guy_(dhmis)"],
        "trigger": ["red guy \\(dhmis\\), don't hug me i'm scared"],
    },
    "eliot_(heroic_ones)": {
        "character": ["eliot_(heroic_ones)"],
        "trigger": ["eliot \\(heroic ones\\), halloween"],
    },
    "starburstsaber_(character)": {
        "character": ["starburstsaber_(character)"],
        "trigger": ["starburstsaber \\(character\\), starburst"],
    },
    "joyride_(colt_quest)": {
        "character": ["joyride_(colt_quest)"],
        "trigger": ["joyride \\(colt quest\\), my little pony"],
    },
    "bergamo": {"character": ["bergamo"], "trigger": ["bergamo, dragon ball"]},
    "muscle_bird": {
        "character": ["muscle_bird"],
        "trigger": ["muscle bird, scottgames"],
    },
    "aegis_(infinitedge)": {
        "character": ["aegis_(infinitedge)"],
        "trigger": ["aegis \\(infinitedge\\), bloodline \\(webcomic\\)"],
    },
    "jake_hart": {"character": ["jake_hart"], "trigger": ["jake hart, book of lust"]},
    "rosy_firefly": {
        "character": ["rosy_firefly"],
        "trigger": ["rosy firefly, my little pony"],
    },
    "sherri_aura": {
        "character": ["sherri_aura"],
        "trigger": ["sherri aura, mythology"],
    },
    "sarek_aran_desian_(character)": {
        "character": ["sarek_aran_desian_(character)"],
        "trigger": ["sarek aran desian \\(character\\), mythology"],
    },
    "zilx_(bugmag)": {
        "character": ["zilx_(bugmag)"],
        "trigger": ["zilx \\(bugmag\\), the binding of isaac \\(series\\)"],
    },
    "sebun_(beastars)": {
        "character": ["sebun_(beastars)"],
        "trigger": ["sebun \\(beastars\\), beastars"],
    },
    "dee_dee_(101_dalmatians)": {
        "character": ["dee_dee_(101_dalmatians)"],
        "trigger": ["dee dee \\(101 dalmatians\\), disney"],
    },
    "max_(maxdamage_rulz)": {
        "character": ["max_(maxdamage_rulz)"],
        "trigger": ["max \\(maxdamage rulz\\), disney"],
    },
    "laxes": {"character": ["laxes"], "trigger": ["laxes, mythology"]},
    "manaka_(aggretsuko)": {
        "character": ["manaka_(aggretsuko)"],
        "trigger": ["manaka \\(aggretsuko\\), sanrio"],
    },
    "unit_04": {"character": ["unit_04"], "trigger": ["unit 04, mythology"]},
    "teo_(world_flipper)": {
        "character": ["teo_(world_flipper)"],
        "trigger": ["teo \\(world flipper\\), cygames"],
    },
    "rai": {"character": ["rai"], "trigger": ["rai, pokemon"]},
    "hershey_the_cat": {
        "character": ["hershey_the_cat"],
        "trigger": ["hershey the cat, sonic the hedgehog \\(series\\)"],
    },
    "usagi_tsukino": {
        "character": ["usagi_tsukino"],
        "trigger": ["usagi tsukino, sailor moon \\(series\\)"],
    },
    "secret_(character)": {
        "character": ["secret_(character)"],
        "trigger": ["secret \\(character\\), mythology"],
    },
    "haley_long": {"character": ["haley_long"], "trigger": ["haley long, disney"]},
    "stimpy_j._cat": {
        "character": ["stimpy_j._cat"],
        "trigger": ["stimpy j. cat, ren and stimpy"],
    },
    "stormtrooper": {
        "character": ["stormtrooper"],
        "trigger": ["stormtrooper, star wars"],
    },
    "ceylon": {"character": ["ceylon"], "trigger": ["ceylon, mythology"]},
    "williamca": {"character": ["williamca"], "trigger": ["williamca, mythology"]},
    "spiky-eared_pichu": {
        "character": ["spiky-eared_pichu"],
        "trigger": ["spiky-eared pichu, pokemon"],
    },
    "chrissy_mccloud": {
        "character": ["chrissy_mccloud"],
        "trigger": ["chrissy mccloud, rascals"],
    },
    "bimbette": {"character": ["bimbette"], "trigger": ["bimbette, warner brothers"]},
    "sophie_(shyguy9)": {
        "character": ["sophie_(shyguy9)"],
        "trigger": ["sophie \\(shyguy9\\), mythology"],
    },
    "nightshade_(kittyprint)": {
        "character": ["nightshade_(kittyprint)"],
        "trigger": ["nightshade \\(kittyprint\\), christmas"],
    },
    "kibacheetah": {
        "character": ["kibacheetah"],
        "trigger": ["kibacheetah, mythology"],
    },
    "ms._peachbottom_(mlp)": {
        "character": ["ms._peachbottom_(mlp)"],
        "trigger": ["ms. peachbottom \\(mlp\\), my little pony"],
    },
    "sibella_dracula": {
        "character": ["sibella_dracula"],
        "trigger": ["sibella dracula, ghoul school"],
    },
    "hawke_(mastergodai)": {
        "character": ["hawke_(mastergodai)"],
        "trigger": ["hawke \\(mastergodai\\), knuckle up!"],
    },
    "thunderbolt_the_chinchilla": {
        "character": ["thunderbolt_the_chinchilla"],
        "trigger": ["thunderbolt the chinchilla, sonic the hedgehog \\(series\\)"],
    },
    "silver_sickle_(oc)": {
        "character": ["silver_sickle_(oc)"],
        "trigger": ["silver sickle \\(oc\\), my little pony"],
    },
    "anise_(quin-nsfw)": {
        "character": ["anise_(quin-nsfw)"],
        "trigger": ["anise \\(quin-nsfw\\), pokemon"],
    },
    "specterdragon": {
        "character": ["specterdragon"],
        "trigger": ["specterdragon, mythology"],
    },
    "rg02_(undertale)": {
        "character": ["rg02_(undertale)"],
        "trigger": ["rg02 \\(undertale\\), undertale \\(series\\)"],
    },
    "martin_mink": {
        "character": ["martin_mink"],
        "trigger": ["martin mink, rutwell forest"],
    },
    "nimbus_whitetail": {
        "character": ["nimbus_whitetail"],
        "trigger": ["nimbus whitetail, mythology"],
    },
    "rosanne_hayes": {
        "character": ["rosanne_hayes"],
        "trigger": ["rosanne hayes, greek mythology"],
    },
    "jamie_knox_(jamiekaboom)": {
        "character": ["jamie_knox_(jamiekaboom)"],
        "trigger": ["jamie knox \\(jamiekaboom\\), mythology"],
    },
    "layla_(mrdirt)": {
        "character": ["layla_(mrdirt)"],
        "trigger": ["layla \\(mrdirt\\), caravan palace"],
    },
    "ferro_the_dragon": {
        "character": ["ferro_the_dragon"],
        "trigger": ["ferro the dragon, mythology"],
    },
    "warfare_alilkira": {
        "character": ["warfare_alilkira"],
        "trigger": ["warfare alilkira, warfare machine"],
    },
    "redshift_(reddrawsstuff)": {
        "character": ["redshift_(reddrawsstuff)"],
        "trigger": ["redshift \\(reddrawsstuff\\), mythology"],
    },
    "jeeper": {"character": ["jeeper"], "trigger": ["jeeper, fetishbruary"]},
    "laika_horse": {
        "character": ["laika_horse"],
        "trigger": ["laika horse, creative commons"],
    },
    "circi_(yobie)": {
        "character": ["circi_(yobie)"],
        "trigger": ["circi \\(yobie\\), pokemon"],
    },
    "monk_(rain_world)": {
        "character": ["monk_(rain_world)"],
        "trigger": ["monk \\(rain world\\), videocult"],
    },
    "vance_(zephyrnok)": {
        "character": ["vance_(zephyrnok)"],
        "trigger": ["vance \\(zephyrnok\\), cool s"],
    },
    "francine_(ruanshi)": {
        "character": ["francine_(ruanshi)"],
        "trigger": ["francine \\(ruanshi\\), mythology"],
    },
    "nikolai_(the_smoke_room)": {
        "character": ["nikolai_(the_smoke_room)"],
        "trigger": ["nikolai \\(the smoke room\\), the smoke room"],
    },
    "judy_(animal_crossing)": {
        "character": ["judy_(animal_crossing)"],
        "trigger": ["judy \\(animal crossing\\), animal crossing"],
    },
    "morgan_(rattfood)": {
        "character": ["morgan_(rattfood)"],
        "trigger": ["morgan \\(rattfood\\), christmas"],
    },
    "bob_(vju79)": {
        "character": ["bob_(vju79)"],
        "trigger": ["bob \\(vju79\\), mythology"],
    },
    "golden_fredina_(cally3d)": {
        "character": ["golden_fredina_(cally3d)"],
        "trigger": ["golden fredina \\(cally3d\\), five nights at freddy's"],
    },
    "golde_(golde)": {
        "character": ["golde_(golde)"],
        "trigger": ["golde \\(golde\\), nintendo"],
    },
    "umbrose": {"character": ["umbrose"], "trigger": ["umbrose, kinktober"]},
    "ashlee_hurwitz": {
        "character": ["ashlee_hurwitz"],
        "trigger": ["ashlee hurwitz, good cheese"],
    },
    "jen_(jindragowolf)": {
        "character": ["jen_(jindragowolf)"],
        "trigger": ["jen \\(jindragowolf\\), mythology"],
    },
    "fievel_mousekewitz": {
        "character": ["fievel_mousekewitz"],
        "trigger": ["fievel mousekewitz, an american tail"],
    },
    "noonie-beyl": {"character": ["noonie-beyl"], "trigger": ["noonie-beyl, nintendo"]},
    "juri_han": {"character": ["juri_han"], "trigger": ["juri han, capcom"]},
    "akali_(lol)": {
        "character": ["akali_(lol)"],
        "trigger": ["akali \\(lol\\), riot games"],
    },
    "vitaly_(madagascar)": {
        "character": ["vitaly_(madagascar)"],
        "trigger": ["vitaly \\(madagascar\\), dreamworks"],
    },
    "ingi_(character)": {
        "character": ["ingi_(character)"],
        "trigger": ["ingi \\(character\\), mythology"],
    },
    "bandana_waddle_dee": {
        "character": ["bandana_waddle_dee"],
        "trigger": ["bandana waddle dee, nintendo"],
    },
    "ingo_(pokemon)": {
        "character": ["ingo_(pokemon)"],
        "trigger": ["ingo \\(pokemon\\), pokemon"],
    },
    "milftwo": {"character": ["milftwo"], "trigger": ["milftwo, pokemon"]},
    "crash_azarel_(character)": {
        "character": ["crash_azarel_(character)"],
        "trigger": ["crash azarel \\(character\\), mythology"],
    },
    "radiance_(mlp)": {
        "character": ["radiance_(mlp)"],
        "trigger": ["radiance \\(mlp\\), my little pony"],
    },
    "bessie_(zp92)": {
        "character": ["bessie_(zp92)"],
        "trigger": ["bessie \\(zp92\\), shirt cut meme"],
    },
    "orthros_(mlp)": {
        "character": ["orthros_(mlp)"],
        "trigger": ["orthros \\(mlp\\), my little pony"],
    },
    "withered_freddy_(fnaf)": {
        "character": ["withered_freddy_(fnaf)"],
        "trigger": ["withered freddy \\(fnaf\\), five nights at freddy's 2"],
    },
    "north_shepherd": {
        "character": ["north_shepherd"],
        "trigger": ["north shepherd, universal studios"],
    },
    "dandee_(character)": {
        "character": ["dandee_(character)"],
        "trigger": ["dandee \\(character\\), my life with fel"],
    },
    "cobb_(shade)": {
        "character": ["cobb_(shade)"],
        "trigger": ["cobb \\(shade\\), greek mythology"],
    },
    "azalia": {"character": ["azalia"], "trigger": ["azalia, mythology"]},
    "recon_scout_teemo": {
        "character": ["recon_scout_teemo"],
        "trigger": ["recon scout teemo, riot games"],
    },
    "azaly": {"character": ["azaly"], "trigger": ["azaly, mythology"]},
    "taylor_vee": {"character": ["taylor_vee"], "trigger": ["taylor vee, nintendo"]},
    "timblackfox_(character)": {
        "character": ["timblackfox_(character)"],
        "trigger": ["timblackfox \\(character\\), mythology"],
    },
    "finn_(phantomfin)": {
        "character": ["finn_(phantomfin)"],
        "trigger": ["finn \\(phantomfin\\), mythology"],
    },
    "haiko_frostypaws": {
        "character": ["haiko_frostypaws"],
        "trigger": ["haiko frostypaws, mythology"],
    },
    "happysheppy": {
        "character": ["happysheppy"],
        "trigger": ["happysheppy, mythology"],
    },
    "dragonmaid_sheou": {
        "character": ["dragonmaid_sheou"],
        "trigger": ["dragonmaid sheou, yu-gi-oh!"],
    },
    "shirano": {"character": ["shirano"], "trigger": ["shirano, cygames"]},
    "tanetomo_(tas)": {
        "character": ["tanetomo_(tas)"],
        "trigger": ["tanetomo \\(tas\\), lifewonders"],
    },
    "amy_(canisfidelis)": {
        "character": ["amy_(canisfidelis)"],
        "trigger": ["amy \\(canisfidelis\\), no nut november"],
    },
    "neco-arc_chaos": {
        "character": ["neco-arc_chaos"],
        "trigger": ["neco-arc chaos, type-moon"],
    },
    "hinoa_(monster_hunter)": {
        "character": ["hinoa_(monster_hunter)"],
        "trigger": ["hinoa \\(monster hunter\\), capcom"],
    },
    "pepper_(crushpepper)": {
        "character": ["pepper_(crushpepper)"],
        "trigger": ["pepper \\(crushpepper\\), nintendo"],
    },
    "meral_fleetfoot": {
        "character": ["meral_fleetfoot"],
        "trigger": ["meral fleetfoot, mythology"],
    },
    "ash_(abz)": {"character": ["ash_(abz)"], "trigger": ["ash \\(abz\\), abz"]},
    "fexa_(cally3d)": {
        "character": ["fexa_(cally3d)"],
        "trigger": ["fexa \\(cally3d\\), scottgames"],
    },
    "gobu": {"character": ["gobu"], "trigger": ["gobu, fair argument \\(meme\\)"]},
    "sierra_(father_of_the_pride)": {
        "character": ["sierra_(father_of_the_pride)"],
        "trigger": ["sierra \\(father of the pride\\), father of the pride"],
    },
    "modo_(bmfm)": {
        "character": ["modo_(bmfm)"],
        "trigger": ["modo \\(bmfm\\), biker mice from mars"],
    },
    "clifford_(red_dog)": {
        "character": ["clifford_(red_dog)"],
        "trigger": ["clifford \\(red dog\\), clifford the big red dog"],
    },
    "ed_(the_lion_king)": {
        "character": ["ed_(the_lion_king)"],
        "trigger": ["ed \\(the lion king\\), disney"],
    },
    "sakuya_izayoi": {
        "character": ["sakuya_izayoi"],
        "trigger": ["sakuya izayoi, touhou"],
    },
    "mr._krabs": {
        "character": ["mr._krabs"],
        "trigger": ["mr. krabs, spongebob squarepants"],
    },
    "hulk": {"character": ["hulk"], "trigger": ["hulk, marvel"]},
    "pumyra": {"character": ["pumyra"], "trigger": ["pumyra, thundercats"]},
    "jay_van_esbroek": {
        "character": ["jay_van_esbroek"],
        "trigger": ["jay van esbroek, my little pony"],
    },
    "artemis_(sailor_moon)": {
        "character": ["artemis_(sailor_moon)"],
        "trigger": ["artemis \\(sailor moon\\), sailor moon \\(series\\)"],
    },
    "pigma_dengar": {
        "character": ["pigma_dengar"],
        "trigger": ["pigma dengar, star fox"],
    },
    "mao_otter": {"character": ["mao_otter"], "trigger": ["mao otter, nintendo"]},
    "duke_nauticus": {
        "character": ["duke_nauticus"],
        "trigger": ["duke nauticus, nintendo"],
    },
    "gunnar's_dad": {
        "character": ["gunnar's_dad"],
        "trigger": ["gunnar's dad, incestaroos"],
    },
    "julian_(animal_crossing)": {
        "character": ["julian_(animal_crossing)"],
        "trigger": ["julian \\(animal crossing\\), animal crossing"],
    },
    "amunet": {"character": ["amunet"], "trigger": ["amunet, christmas"]},
    "van_(sandwich-anomaly)": {
        "character": ["van_(sandwich-anomaly)"],
        "trigger": ["van \\(sandwich-anomaly\\), pokemon"],
    },
    "fawster_(slendid)": {
        "character": ["fawster_(slendid)"],
        "trigger": ["fawster \\(slendid\\), mythology"],
    },
    "navirou": {"character": ["navirou"], "trigger": ["navirou, monster hunter"]},
    "rg01_(undertale)": {
        "character": ["rg01_(undertale)"],
        "trigger": ["rg01 \\(undertale\\), undertale \\(series\\)"],
    },
    "zane_darkpaw": {
        "character": ["zane_darkpaw"],
        "trigger": ["zane darkpaw, mythology"],
    },
    "saphayla_(zelianda)": {
        "character": ["saphayla_(zelianda)"],
        "trigger": ["saphayla \\(zelianda\\), mythology"],
    },
    "landon_(fellowwolf)": {
        "character": ["landon_(fellowwolf)"],
        "trigger": ["landon \\(fellowwolf\\), mythology"],
    },
    "mr._big_(zootopia)": {
        "character": ["mr._big_(zootopia)"],
        "trigger": ["mr. big \\(zootopia\\), disney"],
    },
    "annie_(anaid)": {
        "character": ["annie_(anaid)"],
        "trigger": ["annie \\(anaid\\), thot burger"],
    },
    "marty_(weaver)": {
        "character": ["marty_(weaver)"],
        "trigger": ["marty \\(weaver\\), pack street"],
    },
    "gaghiel": {"character": ["gaghiel"], "trigger": ["gaghiel, pokemon"]},
    "lolbit_(psychojohn2)": {
        "character": ["lolbit_(psychojohn2)"],
        "trigger": ["lolbit \\(psychojohn2\\), scottgames"],
    },
    "will_(hladilnik)": {
        "character": ["will_(hladilnik)"],
        "trigger": ["will \\(hladilnik\\), christmas"],
    },
    "adrian_donovan": {
        "character": ["adrian_donovan"],
        "trigger": ["adrian donovan, the doors"],
    },
    "momo_yaoyorozu": {
        "character": ["momo_yaoyorozu"],
        "trigger": ["momo yaoyorozu, my hero academia"],
    },
    "elim_dorelga": {
        "character": ["elim_dorelga"],
        "trigger": ["elim dorelga, out-of-placers"],
    },
    "dallas_(gingersnaps)": {
        "character": ["dallas_(gingersnaps)"],
        "trigger": ["dallas \\(gingersnaps\\), christmas"],
    },
    "kyra_(fuf)": {"character": ["kyra_(fuf)"], "trigger": ["kyra \\(fuf\\), pokemon"]},
    "kaeli_cedarfallen": {
        "character": ["kaeli_cedarfallen"],
        "trigger": ["kaeli cedarfallen, avatar: the last airbender"],
    },
    "goo_(razy)": {
        "character": ["goo_(razy)"],
        "trigger": ["goo \\(razy\\), subscribestar"],
    },
    "felicia_(tahlian)": {
        "character": ["felicia_(tahlian)"],
        "trigger": ["felicia \\(tahlian\\), blender \\(software\\)"],
    },
    "frosty_(sharky)": {
        "character": ["frosty_(sharky)"],
        "trigger": ["frosty \\(sharky\\), t-shirt/pajamas challenge"],
    },
    "tracy_(linker)": {
        "character": ["tracy_(linker)"],
        "trigger": ["tracy \\(linker\\), pokemon"],
    },
    "lampy_(azura_inalis)": {
        "character": ["lampy_(azura_inalis)"],
        "trigger": ["lampy \\(azura inalis\\), team cherry"],
    },
    "fuko": {"character": ["fuko"], "trigger": ["fuko, dust: an elysian tail"]},
    "amiya_(arknights)": {
        "character": ["amiya_(arknights)"],
        "trigger": ["amiya \\(arknights\\), studio montagne"],
    },
    "ookami_mio": {"character": ["ookami_mio"], "trigger": ["ookami mio, hololive"]},
    "voki_(youwannaslap)": {
        "character": ["voki_(youwannaslap)"],
        "trigger": ["voki \\(youwannaslap\\), mythology"],
    },
    "fuse_(analogpentium)": {
        "character": ["fuse_(analogpentium)"],
        "trigger": ["fuse \\(analogpentium\\), mythology"],
    },
    "alice_the_vixen": {
        "character": ["alice_the_vixen"],
        "trigger": ["alice the vixen, tiny bunny"],
    },
    "pepper_(wonderslug)": {
        "character": ["pepper_(wonderslug)"],
        "trigger": ["pepper \\(wonderslug\\), mythology"],
    },
    "moxxie_(valorlynz)": {
        "character": ["moxxie_(valorlynz)"],
        "trigger": ["moxxie \\(valorlynz\\), helluva boss"],
    },
    "zaush_(zaush)": {
        "character": ["zaush_(zaush)"],
        "trigger": ["zaush \\(zaush\\), kio \\(keovi\\)"],
    },
    "krista_(zillford)": {
        "character": ["krista_(zillford)"],
        "trigger": ["krista \\(zillford\\), disney"],
    },
    "hurst": {
        "character": ["hurst"],
        "trigger": ["hurst, sony interactive entertainment"],
    },
    "rita_(animaniacs)": {
        "character": ["rita_(animaniacs)"],
        "trigger": ["rita \\(animaniacs\\), warner brothers"],
    },
    "mr._whiskers": {
        "character": ["mr._whiskers"],
        "trigger": ["mr. whiskers, disney"],
    },
    "blair_(soul_eater)": {
        "character": ["blair_(soul_eater)"],
        "trigger": ["blair \\(soul eater\\), soul eater"],
    },
    "byzil": {"character": ["byzil"], "trigger": ["byzil, mythology"]},
    "shaytalis": {"character": ["shaytalis"], "trigger": ["shaytalis, fallout"]},
    "bubbles_(powerpuff_girls)": {
        "character": ["bubbles_(powerpuff_girls)"],
        "trigger": ["bubbles \\(powerpuff girls\\), cartoon network"],
    },
    "kintuse": {"character": ["kintuse"], "trigger": ["kintuse, nintendo"]},
    "lilly_(alpha_and_omega)": {
        "character": ["lilly_(alpha_and_omega)"],
        "trigger": ["lilly \\(alpha and omega\\), alpha and omega"],
    },
    "egger": {"character": ["egger"], "trigger": ["egger, mythology"]},
    "shephira_(cert)": {
        "character": ["shephira_(cert)"],
        "trigger": ["shephira \\(cert\\), bunny and fox world"],
    },
    "revoli": {"character": ["revoli"], "trigger": ["revoli"]},
    "hazel_(flittermilk)": {
        "character": ["hazel_(flittermilk)"],
        "trigger": ["hazel \\(flittermilk\\), mythology"],
    },
    "fora": {"character": ["fora"], "trigger": ["fora, mythology"]},
    "adharc": {"character": ["adharc"], "trigger": ["adharc, thorsoneyja"]},
    "arith": {"character": ["arith"], "trigger": ["arith, mythology"]},
    "koslov_(zootopia)": {
        "character": ["koslov_(zootopia)"],
        "trigger": ["koslov \\(zootopia\\), disney"],
    },
    "princess_(paigeforsyth)": {
        "character": ["princess_(paigeforsyth)"],
        "trigger": ["princess \\(paigeforsyth\\), mythology"],
    },
    "ice_wolf_(undertale)": {
        "character": ["ice_wolf_(undertale)"],
        "trigger": ["ice wolf \\(undertale\\), undertale \\(series\\)"],
    },
    "erik_d'javel": {
        "character": ["erik_d'javel"],
        "trigger": ["erik d'javel, to be continued"],
    },
    "bani_(ezpups)": {
        "character": ["bani_(ezpups)"],
        "trigger": ["bani \\(ezpups\\), bani the kitty"],
    },
    "mallow_(pokemon)": {
        "character": ["mallow_(pokemon)"],
        "trigger": ["mallow \\(pokemon\\), pokemon"],
    },
    "the_stag": {
        "character": ["the_stag"],
        "trigger": ["the stag, creatures of the night"],
    },
    "anna_(angels_with_scaly_wings)": {
        "character": ["anna_(angels_with_scaly_wings)"],
        "trigger": ["anna \\(angels with scaly wings\\), angels with scaly wings"],
    },
    "laarx": {"character": ["laarx"], "trigger": ["laarx, pokemon"]},
    "the_devil_(cuphead)": {
        "character": ["the_devil_(cuphead)"],
        "trigger": ["the devil \\(cuphead\\), cuphead \\(game\\)"],
    },
    "omega_wolf_(soulwolven)": {
        "character": ["omega_wolf_(soulwolven)"],
        "trigger": ["omega wolf \\(soulwolven\\), mythology"],
    },
    "cain_(neko3240)": {
        "character": ["cain_(neko3240)"],
        "trigger": ["cain \\(neko3240\\), nintendo"],
    },
    "sapphire_(anglo)": {
        "character": ["sapphire_(anglo)"],
        "trigger": ["sapphire \\(anglo\\), pokemon"],
    },
    "toshi_(kyrosh)": {
        "character": ["toshi_(kyrosh)"],
        "trigger": ["toshi \\(kyrosh\\), curious cat"],
    },
    "yuki_motoe": {
        "character": ["yuki_motoe"],
        "trigger": ["yuki motoe, nintendo switch"],
    },
    "barley_lightfoot": {
        "character": ["barley_lightfoot"],
        "trigger": ["barley lightfoot, disney"],
    },
    "rey_(animatedmau)": {
        "character": ["rey_(animatedmau)"],
        "trigger": ["rey \\(animatedmau\\), christmas"],
    },
    "chloe_(icma)": {
        "character": ["chloe_(icma)"],
        "trigger": ["chloe \\(icma\\), pmd: icma"],
    },
    "dervid_(taktian)": {
        "character": ["dervid_(taktian)"],
        "trigger": ["dervid \\(taktian\\), mythology"],
    },
    "nagi_(nagifur)": {
        "character": ["nagi_(nagifur)"],
        "trigger": ["nagi \\(nagifur\\), halloween"],
    },
    "radley_heeler": {
        "character": ["radley_heeler"],
        "trigger": ["radley heeler, bluey \\(series\\)"],
    },
    "crystal_bloom": {
        "character": ["crystal_bloom"],
        "trigger": ["crystal bloom, mythology"],
    },
    "affax": {"character": ["affax"], "trigger": ["affax, boy kisser \\(meme\\)"]},
    "iono_(pokemon)": {
        "character": ["iono_(pokemon)"],
        "trigger": ["iono \\(pokemon\\), pokemon"],
    },
    "deep_cut_(splatoon)": {
        "character": ["deep_cut_(splatoon)"],
        "trigger": ["deep cut \\(splatoon\\), splatoon"],
    },
    "kiff_chatterley": {
        "character": ["kiff_chatterley"],
        "trigger": ["kiff chatterley, disney"],
    },
    "spade_(tatsuchan18)": {
        "character": ["spade_(tatsuchan18)"],
        "trigger": ["spade \\(tatsuchan18\\), game boy"],
    },
    "sakura_haruno": {
        "character": ["sakura_haruno"],
        "trigger": ["sakura haruno, naruto"],
    },
    "yoko_littner": {
        "character": ["yoko_littner"],
        "trigger": ["yoko littner, tengen toppa gurren lagann"],
    },
    "nefer": {"character": ["nefer"], "trigger": ["nefer, mythology"]},
    "ora_(ora)": {"character": ["ora_(ora)"], "trigger": ["ora \\(ora\\), mythology"]},
    "kero_(cardcaptor_sakura)": {
        "character": ["kero_(cardcaptor_sakura)"],
        "trigger": ["kero \\(cardcaptor sakura\\), cardcaptor sakura"],
    },
    "starscream": {"character": ["starscream"], "trigger": ["starscream, takara tomy"]},
    "stan_(hamtaro)": {
        "character": ["stan_(hamtaro)"],
        "trigger": ["stan \\(hamtaro\\), hamtaro \\(series\\)"],
    },
    "sar": {"character": ["sar"], "trigger": ["sar, mythology"]},
    "milkyway_(truegrave9)": {
        "character": ["milkyway_(truegrave9)"],
        "trigger": ["milkyway \\(truegrave9\\), mythology"],
    },
    "beartato": {"character": ["beartato"], "trigger": ["beartato, mythology"]},
    "robyn_mcclaire": {
        "character": ["robyn_mcclaire"],
        "trigger": ["robyn mcclaire, chimangetsu"],
    },
    "blitzen": {"character": ["blitzen"], "trigger": ["blitzen, christmas"]},
    "manic_the_hedgehog": {
        "character": ["manic_the_hedgehog"],
        "trigger": ["manic the hedgehog, sonic underground"],
    },
    "okono_yuujo": {
        "character": ["okono_yuujo"],
        "trigger": ["okono yuujo, christmas"],
    },
    "duke_(thecon)": {
        "character": ["duke_(thecon)"],
        "trigger": ["duke \\(thecon\\), sonic the hedgehog \\(series\\)"],
    },
    "aella": {"character": ["aella"], "trigger": ["aella, resident evil"]},
    "skan_drake": {
        "character": ["skan_drake"],
        "trigger": ["skan drake, the monster within"],
    },
    "jasmine_ivory": {
        "character": ["jasmine_ivory"],
        "trigger": ["jasmine ivory, scottgames"],
    },
    "travis_(zootopia)": {
        "character": ["travis_(zootopia)"],
        "trigger": ["travis \\(zootopia\\), disney"],
    },
    "drakorax": {"character": ["drakorax"], "trigger": ["drakorax, mythology"]},
    "cindy_(cindyquilava)": {
        "character": ["cindy_(cindyquilava)"],
        "trigger": ["cindy \\(cindyquilava\\), pokemon"],
    },
    "kiri_(sub-res)": {
        "character": ["kiri_(sub-res)"],
        "trigger": ["kiri \\(sub-res\\), touch fluffy tail"],
    },
    "bue_(character)": {
        "character": ["bue_(character)"],
        "trigger": ["bue \\(character\\), samurai jack"],
    },
    "cassidy_(alec8ter)": {
        "character": ["cassidy_(alec8ter)"],
        "trigger": ["cassidy \\(alec8ter\\), disney"],
    },
    "caramel_(cherrikissu)": {
        "character": ["caramel_(cherrikissu)"],
        "trigger": ["caramel \\(cherrikissu\\), christmas"],
    },
    "high_elf_archer_(goblin_slayer)": {
        "character": ["high_elf_archer_(goblin_slayer)"],
        "trigger": ["high elf archer \\(goblin slayer\\), goblin slayer"],
    },
    "dizzy_(101_dalmatians)": {
        "character": ["dizzy_(101_dalmatians)"],
        "trigger": ["dizzy \\(101 dalmatians\\), disney"],
    },
    "lucy_swallows": {
        "character": ["lucy_swallows"],
        "trigger": ["lucy swallows, nintendo"],
    },
    "xi_yue_(tgww)": {
        "character": ["xi_yue_(tgww)"],
        "trigger": ["xi yue \\(tgww\\), the great warrior wall"],
    },
    "cocoline": {"character": ["cocoline"], "trigger": ["cocoline, mythology"]},
    "baozi_(diives)": {
        "character": ["baozi_(diives)"],
        "trigger": ["baozi \\(diives\\), xingzuo temple"],
    },
    "mohinya": {"character": ["mohinya"], "trigger": ["mohinya, twitter"]},
    "kipwolf": {"character": ["kipwolf"], "trigger": ["kipwolf, halloween"]},
    "azazel_(helltaker)": {
        "character": ["azazel_(helltaker)"],
        "trigger": ["azazel \\(helltaker\\), helltaker"],
    },
    "xargos": {"character": ["xargos"], "trigger": ["xargos, mythology"]},
    "cherry_popper": {
        "character": ["cherry_popper"],
        "trigger": ["cherry popper, pokemon"],
    },
    "yano_(odd_taxi)": {
        "character": ["yano_(odd_taxi)"],
        "trigger": ["yano \\(odd taxi\\), odd taxi"],
    },
    "ride_sneasler_(pokemon_legends_arceus)": {
        "character": ["ride_sneasler_(pokemon_legends_arceus)"],
        "trigger": ["ride sneasler \\(pokemon legends arceus\\), pokemon"],
    },
    "camille_(fortnite)": {
        "character": ["camille_(fortnite)"],
        "trigger": ["camille \\(fortnite\\), fortnite"],
    },
    "semple": {"character": ["semple"], "trigger": ["semple, mythology"]},
    "blaze_(wolf)": {
        "character": ["blaze_(wolf)"],
        "trigger": ["blaze \\(wolf\\), my little pony"],
    },
    "splash_woman": {
        "character": ["splash_woman"],
        "trigger": ["splash woman, capcom"],
    },
    "ray_the_flying_squirrel": {
        "character": ["ray_the_flying_squirrel"],
        "trigger": ["ray the flying squirrel, sonic the hedgehog \\(series\\)"],
    },
    "stocking_(pswg)": {
        "character": ["stocking_(pswg)"],
        "trigger": ["stocking \\(pswg\\), panty and stocking with garterbelt"],
    },
    "luther_denholme": {
        "character": ["luther_denholme"],
        "trigger": ["luther denholme, red lantern"],
    },
    "angel_the_catgirl": {
        "character": ["angel_the_catgirl"],
        "trigger": ["angel the catgirl, sonic the hedgehog \\(series\\)"],
    },
    "ed_ryujin": {"character": ["ed_ryujin"], "trigger": ["ed ryujin, mythology"]},
    "kissa_yander": {
        "character": ["kissa_yander"],
        "trigger": ["kissa yander, riot games"],
    },
    "bunnie_(animal_crossing)": {
        "character": ["bunnie_(animal_crossing)"],
        "trigger": ["bunnie \\(animal crossing\\), animal crossing"],
    },
    "gala_(carmessi)": {
        "character": ["gala_(carmessi)"],
        "trigger": ["gala \\(carmessi\\), nintendo"],
    },
    "dizfoley": {"character": ["dizfoley"], "trigger": ["dizfoley, mythology"]},
    "kuugo_(lagoon_lounge)": {
        "character": ["kuugo_(lagoon_lounge)"],
        "trigger": ["kuugo \\(lagoon lounge\\), lagoon lounge"],
    },
    "tabytha_starling": {
        "character": ["tabytha_starling"],
        "trigger": ["tabytha starling, mythology"],
    },
    "willy_(artdecade)": {
        "character": ["willy_(artdecade)"],
        "trigger": ["willy \\(artdecade\\), tumblr"],
    },
    "baymax": {"character": ["baymax"], "trigger": ["baymax, disney"]},
    "jace_zantetsukin": {
        "character": ["jace_zantetsukin"],
        "trigger": ["jace zantetsukin, mythology"],
    },
    "cyote": {"character": ["cyote"], "trigger": ["cyote, sony corporation"]},
    "ozzy_(dingah)": {
        "character": ["ozzy_(dingah)"],
        "trigger": ["ozzy \\(dingah\\), nintendo"],
    },
    "chibi_(c1-11131)": {
        "character": ["chibi_(c1-11131)"],
        "trigger": ["chibi \\(c1-11131\\), pokemon"],
    },
    "quesi": {"character": ["quesi"], "trigger": ["quesi, mythology"]},
    "toy_(mcnasty)": {
        "character": ["toy_(mcnasty)"],
        "trigger": ["toy \\(mcnasty\\), pokemon"],
    },
    "fuf_(character)": {
        "character": ["fuf_(character)"],
        "trigger": ["fuf \\(character\\), heavy metal"],
    },
    "rose_dandy-ba": {
        "character": ["rose_dandy-ba"],
        "trigger": ["rose dandy-ba, mythology"],
    },
    "sven_the_giramon": {
        "character": ["sven_the_giramon"],
        "trigger": ["sven the giramon, digimon"],
    },
    "fate_(trinity-fate62)": {
        "character": ["fate_(trinity-fate62)"],
        "trigger": ["fate \\(trinity-fate62\\), patreon"],
    },
    "kaio_kincaid": {
        "character": ["kaio_kincaid"],
        "trigger": ["kaio kincaid, cartoon network"],
    },
    "fari_paredes": {
        "character": ["fari_paredes"],
        "trigger": ["fari paredes, warcraft"],
    },
    "marion_(changeling_tale)": {
        "character": ["marion_(changeling_tale)"],
        "trigger": ["marion \\(changeling tale\\), changeling tale"],
    },
    "altrue": {"character": ["altrue"], "trigger": ["altrue, mythology"]},
    "bunny_enid": {
        "character": ["bunny_enid"],
        "trigger": ["bunny enid, cartoon network"],
    },
    "tj_hess": {"character": ["tj_hess"], "trigger": ["tj hess, echo \\(game\\)"]},
    "howard_(james_howard)": {
        "character": ["howard_(james_howard)"],
        "trigger": ["howard \\(james howard\\), mythology"],
    },
    "gingy_(gingy_k_fox)": {
        "character": ["gingy_(gingy_k_fox)"],
        "trigger": ["gingy \\(gingy k fox\\), pokemon"],
    },
    "riley_(s-nina)": {
        "character": ["riley_(s-nina)"],
        "trigger": ["riley \\(s-nina\\), pokemon"],
    },
    "nico_(ettechouette)": {
        "character": ["nico_(ettechouette)"],
        "trigger": ["nico \\(ettechouette\\), mythology"],
    },
    "luster_dawn_(mlp)": {
        "character": ["luster_dawn_(mlp)"],
        "trigger": ["luster dawn \\(mlp\\), my little pony"],
    },
    "scp-1471-a_(cumminham)": {
        "character": ["scp-1471-a_(cumminham)"],
        "trigger": ["scp-1471-a \\(cumminham\\), scp foundation"],
    },
    "krowlfer": {"character": ["krowlfer"], "trigger": ["krowlfer, mythology"]},
    "birdvian_(character)": {
        "character": ["birdvian_(character)"],
        "trigger": ["birdvian \\(character\\), nintendo"],
    },
    "mrs.mayberry_(helluva_boss)": {
        "character": ["mrs.mayberry_(helluva_boss)"],
        "trigger": ["mrs.mayberry \\(helluva boss\\), helluva boss"],
    },
    "horse_(centaurworld)": {
        "character": ["horse_(centaurworld)"],
        "trigger": ["horse \\(centaurworld\\), netflix"],
    },
    "dark_the_xenodragon": {
        "character": ["dark_the_xenodragon"],
        "trigger": ["dark the xenodragon, alien \\(franchise\\)"],
    },
    "milfyena": {"character": ["milfyena"], "trigger": ["milfyena, pokemon"]},
    "daisy_(tatsuchan18)": {
        "character": ["daisy_(tatsuchan18)"],
        "trigger": ["daisy \\(tatsuchan18\\), mythology"],
    },
    "diffuse_moose": {
        "character": ["diffuse_moose"],
        "trigger": ["diffuse moose, christmas"],
    },
    "callie_(vinqou)": {
        "character": ["callie_(vinqou)"],
        "trigger": ["callie \\(vinqou\\), caramelldansen"],
    },
    "vixy": {"character": ["vixy"], "trigger": ["vixy, las lindas"]},
    "jon_arbuckle": {
        "character": ["jon_arbuckle"],
        "trigger": ["jon arbuckle, garfield \\(series\\)"],
    },
    "nicolette_the_weasel": {
        "character": ["nicolette_the_weasel"],
        "trigger": ["nicolette the weasel, sonic the hedgehog \\(series\\)"],
    },
    "taross": {"character": ["taross"], "trigger": ["taross, mythology"]},
    "ket_ralus_(character)": {
        "character": ["ket_ralus_(character)"],
        "trigger": ["ket ralus \\(character\\), full frontal frog"],
    },
    "tiggs": {"character": ["tiggs"], "trigger": ["tiggs, las lindas"]},
    "rainbow_mika": {
        "character": ["rainbow_mika"],
        "trigger": ["rainbow mika, capcom"],
    },
    "snagglepuss": {
        "character": ["snagglepuss"],
        "trigger": ["snagglepuss, hanna-barbera"],
    },
    "mary_magdalene": {
        "character": ["mary_magdalene"],
        "trigger": ["mary magdalene, judas and jesus"],
    },
    "stormgryphon": {
        "character": ["stormgryphon"],
        "trigger": ["stormgryphon, mythology"],
    },
    "rosemary_prower": {
        "character": ["rosemary_prower"],
        "trigger": ["rosemary prower, sonic the hedgehog \\(series\\)"],
    },
    "shu_(legendz)": {
        "character": ["shu_(legendz)"],
        "trigger": ["shu \\(legendz\\), legendz"],
    },
    "humphrey": {"character": ["humphrey"], "trigger": ["humphrey, alpha and omega"]},
    "nyama": {"character": ["nyama"], "trigger": ["nyama, mythology"]},
    "amanda_(simonaquarius)": {
        "character": ["amanda_(simonaquarius)"],
        "trigger": ["amanda \\(simonaquarius\\), no nut november"],
    },
    "palace_(character)": {
        "character": ["palace_(character)"],
        "trigger": ["palace \\(character\\), mythology"],
    },
    "valmir": {"character": ["valmir"], "trigger": ["valmir, mythology"]},
    "rannik": {"character": ["rannik"], "trigger": ["rannik, mythology"]},
    "kyoko_usagi": {"character": ["kyoko_usagi"], "trigger": ["kyoko usagi, rascals"]},
    "filia": {"character": ["filia"], "trigger": ["filia, skullgirls"]},
    "neo-spacian_aqua_dolphin": {
        "character": ["neo-spacian_aqua_dolphin"],
        "trigger": ["neo-spacian aqua dolphin, yu-gi-oh!"],
    },
    "drago_(bakugan)": {
        "character": ["drago_(bakugan)"],
        "trigger": ["drago \\(bakugan\\), bakugan"],
    },
    "dark_gaia": {
        "character": ["dark_gaia"],
        "trigger": ["dark gaia, sonic the hedgehog \\(series\\)"],
    },
    "gear_(mlp)": {
        "character": ["gear_(mlp)"],
        "trigger": ["gear \\(mlp\\), my little pony"],
    },
    "jb_greymane": {"character": ["jb_greymane"], "trigger": ["jb greymane, pokemon"]},
    "chocolate_chips_(oc)": {
        "character": ["chocolate_chips_(oc)"],
        "trigger": ["chocolate chips \\(oc\\), my little pony"],
    },
    "hilary_(regular_show)": {
        "character": ["hilary_(regular_show)"],
        "trigger": ["hilary \\(regular show\\), cartoon network"],
    },
    "deadbeat": {"character": ["deadbeat"], "trigger": ["deadbeat, sega"]},
    "kara_(trippledot)": {
        "character": ["kara_(trippledot)"],
        "trigger": ["kara \\(trippledot\\), mythology"],
    },
    "igraine": {"character": ["igraine"], "trigger": ["igraine, christmas"]},
    "styx_y._renegade": {
        "character": ["styx_y._renegade"],
        "trigger": ["styx y. renegade, mythology"],
    },
    "alhazred_(ralek)": {
        "character": ["alhazred_(ralek)"],
        "trigger": ["alhazred \\(ralek\\), my little pony"],
    },
    "scarlet_sound_(oc)": {
        "character": ["scarlet_sound_(oc)"],
        "trigger": ["scarlet sound \\(oc\\), my little pony"],
    },
    "cookie_(critterclaws)": {
        "character": ["cookie_(critterclaws)"],
        "trigger": ["cookie \\(critterclaws\\), mythology"],
    },
    "princess_(nicoya)": {
        "character": ["princess_(nicoya)"],
        "trigger": ["princess \\(nicoya\\), christmas"],
    },
    "neko_hakase": {
        "character": ["neko_hakase"],
        "trigger": ["neko hakase, cat busters"],
    },
    "elise_(sousuke81)": {
        "character": ["elise_(sousuke81)"],
        "trigger": ["elise \\(sousuke81\\), mythology"],
    },
    "audrey_(lizet)": {
        "character": ["audrey_(lizet)"],
        "trigger": ["audrey \\(lizet\\), mythology"],
    },
    "claire_(bunnybits)": {
        "character": ["claire_(bunnybits)"],
        "trigger": ["claire \\(bunnybits\\), nintendo"],
    },
    "koorivlf_tycoon": {
        "character": ["koorivlf_tycoon"],
        "trigger": ["koorivlf tycoon, mythology"],
    },
    "selena_(baelfire117)": {
        "character": ["selena_(baelfire117)"],
        "trigger": ["selena \\(baelfire117\\), mythology"],
    },
    "deja_vu_(101_dalmatians)": {
        "character": ["deja_vu_(101_dalmatians)"],
        "trigger": ["deja vu \\(101 dalmatians\\), disney"],
    },
    "jay_(fizzyjay)": {
        "character": ["jay_(fizzyjay)"],
        "trigger": ["jay \\(fizzyjay\\), nintendo"],
    },
    "malphas_(enginetrap)": {
        "character": ["malphas_(enginetrap)"],
        "trigger": ["malphas \\(enginetrap\\), mythology"],
    },
    "nikki_(saucy)": {
        "character": ["nikki_(saucy)"],
        "trigger": ["nikki \\(saucy\\), patreon"],
    },
    "cliff_(unpopularwolf)": {
        "character": ["cliff_(unpopularwolf)"],
        "trigger": ["cliff \\(unpopularwolf\\), paradisebear"],
    },
    "hyouza": {"character": ["hyouza"], "trigger": ["hyouza, patreon"]},
    "shou_(securipun)": {
        "character": ["shou_(securipun)"],
        "trigger": ["shou \\(securipun\\), mythology"],
    },
    "ace_(tuftydoggo)": {
        "character": ["ace_(tuftydoggo)"],
        "trigger": ["ace \\(tuftydoggo\\), pokemon"],
    },
    "dart_(brok_the_investigator)": {
        "character": ["dart_(brok_the_investigator)"],
        "trigger": ["dart \\(brok the investigator\\), brok the investigator"],
    },
    "remi_(neo_geppetto)": {
        "character": ["remi_(neo_geppetto)"],
        "trigger": ["remi \\(neo geppetto\\), patreon"],
    },
    "tindalos_(tas)": {
        "character": ["tindalos_(tas)"],
        "trigger": ["tindalos \\(tas\\), lifewonders"],
    },
    "korwin": {"character": ["korwin"], "trigger": ["korwin, riot games"]},
    "alma_(capaoculta)": {
        "character": ["alma_(capaoculta)"],
        "trigger": ["alma \\(capaoculta\\), ko-fi"],
    },
    "borealis_(live_a_hero)": {
        "character": ["borealis_(live_a_hero)"],
        "trigger": ["borealis \\(live a hero\\), lifewonders"],
    },
    "raeford_burke": {
        "character": ["raeford_burke"],
        "trigger": ["raeford burke, brogulls"],
    },
    "senri_ooedo": {"character": ["senri_ooedo"], "trigger": ["senri ooedo, vocaloid"]},
    "alicia_(domibun)": {
        "character": ["alicia_(domibun)"],
        "trigger": ["alicia \\(domibun\\), source filmmaker"],
    },
    "heather_(over_the_hedge)": {
        "character": ["heather_(over_the_hedge)"],
        "trigger": ["heather \\(over the hedge\\), over the hedge"],
    },
    "karmakat": {"character": ["karmakat"], "trigger": ["karmakat, swat kats"]},
    "drake_mallard": {
        "character": ["drake_mallard"],
        "trigger": ["drake mallard, disney"],
    },
    "prince_john": {"character": ["prince_john"], "trigger": ["prince john, disney"]},
    "tre_(milligram_smile)": {
        "character": ["tre_(milligram_smile)"],
        "trigger": ["tre \\(milligram smile\\), nintendo"],
    },
    "yoda": {"character": ["yoda"], "trigger": ["yoda, star wars"]},
    "salem_(sutherncross2006)": {
        "character": ["salem_(sutherncross2006)"],
        "trigger": ["salem \\(sutherncross2006\\), mythology"],
    },
    "caninius_dog": {
        "character": ["caninius_dog"],
        "trigger": ["caninius dog, catdog \\(series\\)"],
    },
    "lobo_(animal_crossing)": {
        "character": ["lobo_(animal_crossing)"],
        "trigger": ["lobo \\(animal crossing\\), animal crossing"],
    },
    "samurai_jack_(character)": {
        "character": ["samurai_jack_(character)"],
        "trigger": ["samurai jack \\(character\\), samurai jack"],
    },
    "sanlich": {"character": ["sanlich"], "trigger": ["sanlich, mythology"]},
    "bunnie_love_(character)": {
        "character": ["bunnie_love_(character)"],
        "trigger": ["bunnie love \\(character\\), valve"],
    },
    "tsutami": {"character": ["tsutami"], "trigger": ["tsutami, pokemon"]},
    "bimbo_bear": {
        "character": ["bimbo_bear"],
        "trigger": ["bimbo bear, bimbo \\(bakery\\)"],
    },
    "the_sake_ninja": {
        "character": ["the_sake_ninja"],
        "trigger": ["the sake ninja, thesakeninja"],
    },
    "peter_quill": {"character": ["peter_quill"], "trigger": ["peter quill, marvel"]},
    "seashell_(canisfidelis)": {
        "character": ["seashell_(canisfidelis)"],
        "trigger": ["seashell \\(canisfidelis\\), nintendo"],
    },
    "coal_(catastrophe)": {
        "character": ["coal_(catastrophe)"],
        "trigger": ["coal \\(catastrophe\\), piper perri surrounded"],
    },
    "red_(glopossum)": {
        "character": ["red_(glopossum)"],
        "trigger": ["red \\(glopossum\\), mythology"],
    },
    "elisa_(maddeku)": {
        "character": ["elisa_(maddeku)"],
        "trigger": ["elisa \\(maddeku\\), family guy death pose"],
    },
    "espy_(yo-kai_watch)": {
        "character": ["espy_(yo-kai_watch)"],
        "trigger": ["espy \\(yo-kai watch\\), yo-kai watch"],
    },
    "sharla_(zootopia)": {
        "character": ["sharla_(zootopia)"],
        "trigger": ["sharla \\(zootopia\\), disney"],
    },
    "puff_(softestpuffss)": {
        "character": ["puff_(softestpuffss)"],
        "trigger": ["puff \\(softestpuffss\\), nintendo"],
    },
    "gabe_(james_howard)": {
        "character": ["gabe_(james_howard)"],
        "trigger": ["gabe \\(james howard\\), patreon"],
    },
    "etrius_van_randr": {
        "character": ["etrius_van_randr"],
        "trigger": ["etrius van randr, sigma au"],
    },
    "hulooo": {"character": ["hulooo"], "trigger": ["hulooo, mythology"]},
    "maggie_(kitty_bobo)": {
        "character": ["maggie_(kitty_bobo)"],
        "trigger": ["maggie \\(kitty bobo\\), a kitty bobo show"],
    },
    "hakumen_(tas)": {
        "character": ["hakumen_(tas)"],
        "trigger": ["hakumen \\(tas\\), lifewonders"],
    },
    "pizza_pup": {"character": ["pizza_pup"], "trigger": ["pizza pup, pokemon"]},
    "luna_(zummeng)": {
        "character": ["luna_(zummeng)"],
        "trigger": ["luna \\(zummeng\\), mythology"],
    },
    "cassandra_hart": {
        "character": ["cassandra_hart"],
        "trigger": ["cassandra hart, book of lust"],
    },
    "ophelia_(sssonic2)": {
        "character": ["ophelia_(sssonic2)"],
        "trigger": ["ophelia \\(sssonic2\\), mythology"],
    },
    "lucy_(wherewolf)": {
        "character": ["lucy_(wherewolf)"],
        "trigger": ["lucy \\(wherewolf\\), pixiv fanbox"],
    },
    "tanner_(mao_mao)": {
        "character": ["tanner_(mao_mao)"],
        "trigger": ["tanner \\(mao mao\\), cartoon network"],
    },
    "dallas_prairiewind": {
        "character": ["dallas_prairiewind"],
        "trigger": ["dallas prairiewind, mythology"],
    },
    "meru_(biggreen)": {
        "character": ["meru_(biggreen)"],
        "trigger": ["meru \\(biggreen\\), mythology"],
    },
    "liz_(biggreen)": {
        "character": ["liz_(biggreen)"],
        "trigger": ["liz \\(biggreen\\), mythology"],
    },
    "lolo_(lomidepuzlo)": {
        "character": ["lolo_(lomidepuzlo)"],
        "trigger": ["lolo \\(lomidepuzlo\\), mythology"],
    },
    "riley_(giidenuts)": {
        "character": ["riley_(giidenuts)"],
        "trigger": ["riley \\(giidenuts\\), no nut november"],
    },
    "mia_(eag1e)": {
        "character": ["mia_(eag1e)"],
        "trigger": ["mia \\(eag1e\\), mythology"],
    },
    "pastel_(bigcozyorca)": {
        "character": ["pastel_(bigcozyorca)"],
        "trigger": ["pastel \\(bigcozyorca\\), nintendo"],
    },
    "chaz_(helluva_boss)": {
        "character": ["chaz_(helluva_boss)"],
        "trigger": ["chaz \\(helluva boss\\), helluva boss"],
    },
    "elephant_peach": {
        "character": ["elephant_peach"],
        "trigger": ["elephant peach, mario bros"],
    },
    "draco_(dragonheart)": {
        "character": ["draco_(dragonheart)"],
        "trigger": ["draco \\(dragonheart\\), universal studios"],
    },
    "dynamite_(kadath)": {
        "character": ["dynamite_(kadath)"],
        "trigger": ["dynamite \\(kadath\\), patreon"],
    },
    "modem_(character)": {
        "character": ["modem_(character)"],
        "trigger": ["modem \\(character\\), mythology"],
    },
    "karin_(tetetor-oort)": {
        "character": ["karin_(tetetor-oort)"],
        "trigger": ["karin \\(tetetor-oort\\), mythology"],
    },
    "adolf_hitler": {
        "character": ["adolf_hitler"],
        "trigger": ["adolf hitler, real world"],
    },
    "moomintroll": {
        "character": ["moomintroll"],
        "trigger": ["moomintroll, the moomins"],
    },
    "captain_america": {
        "character": ["captain_america"],
        "trigger": ["captain america, marvel"],
    },
    "ryu_(street_fighter)": {
        "character": ["ryu_(street_fighter)"],
        "trigger": ["ryu \\(street fighter\\), capcom"],
    },
    "phyco": {"character": ["phyco"], "trigger": ["phyco, nintendo"]},
    "tragia": {
        "character": ["tragia"],
        "trigger": ["tragia, the secret lives of flowers"],
    },
    "nemi": {"character": ["nemi"], "trigger": ["nemi, mythology"]},
    "miss_fortune_(lol)": {
        "character": ["miss_fortune_(lol)"],
        "trigger": ["miss fortune \\(lol\\), riot games"],
    },
    "genn_greymane": {
        "character": ["genn_greymane"],
        "trigger": ["genn greymane, warcraft"],
    },
    "boss_wolf": {"character": ["boss_wolf"], "trigger": ["boss wolf, kung fu panda"]},
    "nellko": {"character": ["nellko"], "trigger": ["nellko, ctenophorae"]},
    "captain_otter": {
        "character": ["captain_otter"],
        "trigger": ["captain otter, mythology"],
    },
    "happy_(fairy_tail)": {
        "character": ["happy_(fairy_tail)"],
        "trigger": ["happy \\(fairy tail\\), fairy tail"],
    },
    "attea": {"character": ["attea"], "trigger": ["attea, cartoon network"]},
    "paintheart": {
        "character": ["paintheart"],
        "trigger": ["paintheart, my little pony"],
    },
    "shantae_(monkey_form)": {
        "character": ["shantae_(monkey_form)"],
        "trigger": ["shantae \\(monkey form\\), wayforward"],
    },
    "ellie_(tlou)": {
        "character": ["ellie_(tlou)"],
        "trigger": ["ellie \\(tlou\\), the last of us"],
    },
    "leon_henderson": {
        "character": ["leon_henderson"],
        "trigger": ["leon henderson, fender musical instruments corporation"],
    },
    "lylla": {"character": ["lylla"], "trigger": ["lylla, marvel"]},
    "sureibu": {"character": ["sureibu"], "trigger": ["sureibu, my little pony"]},
    "skynex_(rajak)": {
        "character": ["skynex_(rajak)"],
        "trigger": ["skynex \\(rajak\\), pokemon"],
    },
    "mitty": {"character": ["mitty"], "trigger": ["mitty, made in abyss"]},
    "turkinwif": {"character": ["turkinwif"], "trigger": ["turkinwif, mythology"]},
    "dogaressa": {
        "character": ["dogaressa"],
        "trigger": ["dogaressa, undertale \\(series\\)"],
    },
    "bonfire_(buttocher)": {
        "character": ["bonfire_(buttocher)"],
        "trigger": ["bonfire \\(buttocher\\), mythology"],
    },
    "duck_guy_(dhmis)": {
        "character": ["duck_guy_(dhmis)"],
        "trigger": ["duck guy \\(dhmis\\), don't hug me i'm scared"],
    },
    "nate_(mindnomad)": {
        "character": ["nate_(mindnomad)"],
        "trigger": ["nate \\(mindnomad\\), sega"],
    },
    "mono_(badgeroo)": {
        "character": ["mono_(badgeroo)"],
        "trigger": ["mono \\(badgeroo\\), mythology"],
    },
    "lone_(lonewolffl)": {
        "character": ["lone_(lonewolffl)"],
        "trigger": ["lone \\(lonewolffl\\), pokemon"],
    },
    "zack_(naruever)": {
        "character": ["zack_(naruever)"],
        "trigger": ["zack \\(naruever\\), patreon"],
    },
    "pointedfox_(character)": {
        "character": ["pointedfox_(character)"],
        "trigger": ["pointedfox \\(character\\), disney"],
    },
    "josh_oliver": {"character": ["josh_oliver"], "trigger": ["josh oliver, texnatsu"]},
    "anai_(aggretsuko)": {
        "character": ["anai_(aggretsuko)"],
        "trigger": ["anai \\(aggretsuko\\), sanrio"],
    },
    "kiara_aman": {"character": ["kiara_aman"], "trigger": ["kiara aman, mythology"]},
    "emil_(funkybun)": {
        "character": ["emil_(funkybun)"],
        "trigger": ["emil \\(funkybun\\), halloween"],
    },
    "ash_cinder": {"character": ["ash_cinder"], "trigger": ["ash cinder, nintendo"]},
    "dou_(diives)": {
        "character": ["dou_(diives)"],
        "trigger": ["dou \\(diives\\), xingzuo temple"],
    },
    "silvius_(draethon)": {
        "character": ["silvius_(draethon)"],
        "trigger": ["silvius \\(draethon\\), disney"],
    },
    "rotto_(mrrottson)": {
        "character": ["rotto_(mrrottson)"],
        "trigger": ["rotto \\(mrrottson\\), nintendo"],
    },
    "alcitron": {"character": ["alcitron"], "trigger": ["alcitron, mythology"]},
    "dylan_ramos": {
        "character": ["dylan_ramos"],
        "trigger": ["dylan ramos, the human heart \\(game\\)"],
    },
    "samantha_(snoot_game)": {
        "character": ["samantha_(snoot_game)"],
        "trigger": ["samantha \\(snoot game\\), cavemanon studios"],
    },
    "hydoor": {"character": ["hydoor"], "trigger": ["hydoor, lifewonders"]},
    "goliath_deathclaw_(subbyclaw)": {
        "character": ["goliath_deathclaw_(subbyclaw)"],
        "trigger": ["goliath deathclaw \\(subbyclaw\\), fallout"],
    },
    "lili_(umbraunderscore)": {
        "character": ["lili_(umbraunderscore)"],
        "trigger": ["lili \\(umbraunderscore\\), no nut november"],
    },
    "frinn": {"character": ["frinn"], "trigger": ["frinn, christmas"]},
    "tiger_(petruz)": {
        "character": ["tiger_(petruz)"],
        "trigger": ["tiger \\(petruz\\), petruz \\(copyright\\)"],
    },
    "syrup_(kumalino)": {
        "character": ["syrup_(kumalino)"],
        "trigger": ["syrup \\(kumalino\\), kumalino"],
    },
    "brooklyn_hayes": {
        "character": ["brooklyn_hayes"],
        "trigger": ["brooklyn hayes, hasbro"],
    },
    "mink_(tatsuchan18)": {
        "character": ["mink_(tatsuchan18)"],
        "trigger": ["mink \\(tatsuchan18\\), mythology"],
    },
    "rusty_(behind_the_lens)": {
        "character": ["rusty_(behind_the_lens)"],
        "trigger": ["rusty \\(behind the lens\\), behind the lens"],
    },
    "ramzyuu_(ramzyru)": {
        "character": ["ramzyuu_(ramzyru)"],
        "trigger": ["ramzyuu \\(ramzyru\\), mythology"],
    },
    "hsien-ko_(darkstalkers)": {
        "character": ["hsien-ko_(darkstalkers)"],
        "trigger": ["hsien-ko \\(darkstalkers\\), darkstalkers"],
    },
    "butch_(cursedmarked)": {
        "character": ["butch_(cursedmarked)"],
        "trigger": ["butch \\(cursedmarked\\), i.d.e.k.a"],
    },
    "blossom_(powerpuff_girls)": {
        "character": ["blossom_(powerpuff_girls)"],
        "trigger": ["blossom \\(powerpuff girls\\), cartoon network"],
    },
    "akamaru": {"character": ["akamaru"], "trigger": ["akamaru, naruto"]},
    "pulsar_(character)": {
        "character": ["pulsar_(character)"],
        "trigger": ["pulsar \\(character\\), mythology"],
    },
    "skrien": {"character": ["skrien"], "trigger": ["skrien, pokemon"]},
    "rezzic": {"character": ["rezzic"], "trigger": ["rezzic, dnd homebrew"]},
    "andross": {"character": ["andross"], "trigger": ["andross, star fox"]},
    "kieri_suizahn": {
        "character": ["kieri_suizahn"],
        "trigger": ["kieri suizahn, slightly damned"],
    },
    "jayjay": {"character": ["jayjay"], "trigger": ["jayjay, naruto"]},
    "alex_dowski": {"character": ["alex_dowski"], "trigger": ["alex dowski, dowski"]},
    "ornifex": {"character": ["ornifex"], "trigger": ["ornifex, fromsoftware"]},
    "soren_ashe": {"character": ["soren_ashe"], "trigger": ["soren ashe, mythology"]},
    "tasha_(animal_crossing)": {
        "character": ["tasha_(animal_crossing)"],
        "trigger": ["tasha \\(animal crossing\\), animal crossing"],
    },
    "aria_(neracoda)": {
        "character": ["aria_(neracoda)"],
        "trigger": ["aria \\(neracoda\\), mythology"],
    },
    "pluvian": {"character": ["pluvian"], "trigger": ["pluvian, mythology"]},
    "tora_chitose": {
        "character": ["tora_chitose"],
        "trigger": ["tora chitose, mythology"],
    },
    "krunch_the_kremling": {
        "character": ["krunch_the_kremling"],
        "trigger": ["krunch the kremling, diddy kong racing"],
    },
    "syl_(fvt)": {
        "character": ["syl_(fvt)"],
        "trigger": ["syl \\(fvt\\), fairies vs tentacles"],
    },
    "solicia": {"character": ["solicia"], "trigger": ["solicia, mythology"]},
    "yennefer": {"character": ["yennefer"], "trigger": ["yennefer, the witcher"]},
    "gemini_the_otter": {
        "character": ["gemini_the_otter"],
        "trigger": ["gemini the otter, mythology"],
    },
    "lapis_(mellonsoda)": {
        "character": ["lapis_(mellonsoda)"],
        "trigger": ["lapis \\(mellonsoda\\), nintendo"],
    },
    "null_(sssonic2)": {
        "character": ["null_(sssonic2)"],
        "trigger": ["null \\(sssonic2\\), christmas"],
    },
    "snowy_(creatures_of_the_night)": {
        "character": ["snowy_(creatures_of_the_night)"],
        "trigger": ["snowy \\(creatures of the night\\), creatures of the night"],
    },
    "marie_(angstrom)": {
        "character": ["marie_(angstrom)"],
        "trigger": ["marie \\(angstrom\\), pokemon"],
    },
    "candy_borowski": {
        "character": ["candy_borowski"],
        "trigger": ["candy borowski, night in the woods"],
    },
    "fen_(lagotrope)": {
        "character": ["fen_(lagotrope)"],
        "trigger": ["fen \\(lagotrope\\), tgchan"],
    },
    "texi_(yitexity)": {
        "character": ["texi_(yitexity)"],
        "trigger": ["texi \\(yitexity\\), disney"],
    },
    "matemi_(matemi)": {
        "character": ["matemi_(matemi)"],
        "trigger": ["matemi \\(matemi\\), pokemon"],
    },
    "ripper_(jurassic_world)": {
        "character": ["ripper_(jurassic_world)"],
        "trigger": ["ripper \\(jurassic world\\), universal studios"],
    },
    "wisp_(warframe)": {
        "character": ["wisp_(warframe)"],
        "trigger": ["wisp \\(warframe\\), warframe"],
    },
    "lunares_(freckles)": {
        "character": ["lunares_(freckles)"],
        "trigger": ["lunares \\(freckles\\), my little pony"],
    },
    "rozick": {"character": ["rozick"], "trigger": ["rozick, pokemon"]},
    "peach_(peachymewtwo)": {
        "character": ["peach_(peachymewtwo)"],
        "trigger": ["peach \\(peachymewtwo\\), pokemon"],
    },
    "varan_(evilsx)": {
        "character": ["varan_(evilsx)"],
        "trigger": ["varan \\(evilsx\\), my little pony"],
    },
    "glendale_(centaurworld)": {
        "character": ["glendale_(centaurworld)"],
        "trigger": ["glendale \\(centaurworld\\), netflix"],
    },
    "shaya_(dalwart)": {
        "character": ["shaya_(dalwart)"],
        "trigger": ["shaya \\(dalwart\\), patreon"],
    },
    "michael_(zourik)": {
        "character": ["michael_(zourik)"],
        "trigger": ["michael \\(zourik\\), pokemon"],
    },
    "christopher_wyvern": {
        "character": ["christopher_wyvern"],
        "trigger": ["christopher wyvern, mythology"],
    },
    "changle_(ffjjfjci)": {
        "character": ["changle_(ffjjfjci)"],
        "trigger": ["changle \\(ffjjfjci\\), pocky"],
    },
    "pozole_(character)": {
        "character": ["pozole_(character)"],
        "trigger": ["pozole \\(character\\), nintendo"],
    },
    "yukon": {"character": ["yukon"], "trigger": ["yukon, bluey \\(series\\)"]},
    "ripley_(snoot_game)": {
        "character": ["ripley_(snoot_game)"],
        "trigger": ["ripley \\(snoot game\\), cavemanon studios"],
    },
    "grove_(regalbuster)": {
        "character": ["grove_(regalbuster)"],
        "trigger": ["grove \\(regalbuster\\), grove \\(game\\)"],
    },
    "jimnsei_(character)": {
        "character": ["jimnsei_(character)"],
        "trigger": ["jimnsei \\(character\\), monster energy"],
    },
    "drip_(kumalino)": {
        "character": ["drip_(kumalino)"],
        "trigger": ["drip \\(kumalino\\), kumalino"],
    },
    "hiker_(thepatchedragon)": {
        "character": ["hiker_(thepatchedragon)"],
        "trigger": ["hiker \\(thepatchedragon\\), mythology"],
    },
    "baal_(cult_of_the_lamb)": {
        "character": ["baal_(cult_of_the_lamb)"],
        "trigger": ["baal \\(cult of the lamb\\), cult of the lamb"],
    },
    "lps_339": {"character": ["lps_339"], "trigger": ["lps 339, hasbro"]},
    "ko_(fabio_paulino)": {
        "character": ["ko_(fabio_paulino)"],
        "trigger": ["ko \\(fabio paulino\\), batting the ball"],
    },
    "tirrel_(tirrel)": {
        "character": ["tirrel_(tirrel)"],
        "trigger": ["tirrel \\(tirrel\\), mythology"],
    },
    "chell": {"character": ["chell"], "trigger": ["chell, valve"]},
    "fleki_(character)": {
        "character": ["fleki_(character)"],
        "trigger": ["fleki \\(character\\), mythology"],
    },
    "skippy_squirrel": {
        "character": ["skippy_squirrel"],
        "trigger": ["skippy squirrel, warner brothers"],
    },
    "yin_(yin_yang_yo!)": {
        "character": ["yin_(yin_yang_yo!)"],
        "trigger": ["yin \\(yin yang yo!\\), yin yang yo!"],
    },
    "bernadette_hedgehog": {
        "character": ["bernadette_hedgehog"],
        "trigger": ["bernadette hedgehog, sonic the hedgehog \\(series\\)"],
    },
    "kitsu": {"character": ["kitsu"], "trigger": ["kitsu, mythology"]},
    "amaranth": {"character": ["amaranth"], "trigger": ["amaranth, mythology"]},
    "pom_pom_(mario)": {
        "character": ["pom_pom_(mario)"],
        "trigger": ["pom pom \\(mario\\), mario bros"],
    },
    "chris_(teckly)": {
        "character": ["chris_(teckly)"],
        "trigger": ["chris \\(teckly\\), mythology"],
    },
    "shauna_(pokemon)": {
        "character": ["shauna_(pokemon)"],
        "trigger": ["shauna \\(pokemon\\), pokemon"],
    },
    "poppy_(animal_crossing)": {
        "character": ["poppy_(animal_crossing)"],
        "trigger": ["poppy \\(animal crossing\\), animal crossing"],
    },
    "saffron_(safurantora)": {
        "character": ["saffron_(safurantora)"],
        "trigger": ["saffron \\(safurantora\\), jojo's bizarre adventure"],
    },
    "rui_(sugaru)": {
        "character": ["rui_(sugaru)"],
        "trigger": ["rui \\(sugaru\\), pokemon"],
    },
    "photoshop_flowey": {
        "character": ["photoshop_flowey"],
        "trigger": ["photoshop flowey, undertale \\(series\\)"],
    },
    "kitty_(under(her)tail)": {
        "character": ["kitty_(under(her)tail)"],
        "trigger": ["kitty \\(under(her)tail\\), undertale \\(series\\)"],
    },
    "kippy_the_sharkip": {
        "character": ["kippy_the_sharkip"],
        "trigger": ["kippy the sharkip, pokemon"],
    },
    "bianca_(sheep_and_wolves)": {
        "character": ["bianca_(sheep_and_wolves)"],
        "trigger": ["bianca \\(sheep and wolves\\), sheep and wolves"],
    },
    "asuri_(brawlhalla)": {
        "character": ["asuri_(brawlhalla)"],
        "trigger": ["asuri \\(brawlhalla\\), brawlhalla"],
    },
    "amelia_abernachy": {
        "character": ["amelia_abernachy"],
        "trigger": ["amelia abernachy, pokemon"],
    },
    "ozzy_(weaver)": {
        "character": ["ozzy_(weaver)"],
        "trigger": ["ozzy \\(weaver\\), pack street"],
    },
    "july_hopps_(mistermead)": {
        "character": ["july_hopps_(mistermead)"],
        "trigger": ["july hopps \\(mistermead\\), disney"],
    },
    "holly_applebee": {
        "character": ["holly_applebee"],
        "trigger": ["holly applebee, halloween"],
    },
    "wingman": {
        "character": ["wingman"],
        "trigger": ["wingman, creatures of the night"],
    },
    "muscle_mouse": {
        "character": ["muscle_mouse"],
        "trigger": ["muscle mouse, meme clothing"],
    },
    "stella_(aoino)": {
        "character": ["stella_(aoino)"],
        "trigger": ["stella \\(aoino\\), pocky"],
    },
    "nantangitan": {
        "character": ["nantangitan"],
        "trigger": ["nantangitan, mythology"],
    },
    "miralle": {"character": ["miralle"], "trigger": ["miralle, patreon"]},
    "catty_baby": {
        "character": ["catty_baby"],
        "trigger": ["catty baby, tito lizzardo and catty b"],
    },
    "femboi_lugia_(lightningfire12)": {
        "character": ["femboi_lugia_(lightningfire12)"],
        "trigger": ["femboi lugia \\(lightningfire12\\), pokemon"],
    },
    "cider_(ciderbunart)": {
        "character": ["cider_(ciderbunart)"],
        "trigger": ["cider \\(ciderbunart\\), disney"],
    },
    "spiritpaw": {"character": ["spiritpaw"], "trigger": ["spiritpaw, mythology"]},
    "samantha_reyes": {
        "character": ["samantha_reyes"],
        "trigger": ["samantha reyes, mythology"],
    },
    "lodi_(character)": {
        "character": ["lodi_(character)"],
        "trigger": ["lodi \\(character\\), mythology"],
    },
    "julia_(apizzatrash)": {
        "character": ["julia_(apizzatrash)"],
        "trigger": ["julia \\(apizzatrash\\), mythology"],
    },
    "kaneinu_kosuke": {
        "character": ["kaneinu_kosuke"],
        "trigger": ["kaneinu kosuke, to be continued"],
    },
    "loki_(lowkeytoast)": {
        "character": ["loki_(lowkeytoast)"],
        "trigger": ["loki \\(lowkeytoast\\), mythology"],
    },
    "minoto_the_hub_maiden": {
        "character": ["minoto_the_hub_maiden"],
        "trigger": ["minoto the hub maiden, capcom"],
    },
    "heron_stellanimus": {
        "character": ["heron_stellanimus"],
        "trigger": ["heron stellanimus, videocult"],
    },
    "ryuji_kaiyo": {"character": ["ryuji_kaiyo"], "trigger": ["ryuji kaiyo, pokemon"]},
    "blu_(bludoe)": {
        "character": ["blu_(bludoe)"],
        "trigger": ["blu \\(bludoe\\), nintendo"],
    },
    "mouse_(maynara)": {
        "character": ["mouse_(maynara)"],
        "trigger": ["mouse \\(maynara\\), in front of my salad"],
    },
    "kiriko_(overwatch)": {
        "character": ["kiriko_(overwatch)"],
        "trigger": ["kiriko \\(overwatch\\), overwatch"],
    },
    "papa_bear_(puss_in_boots)": {
        "character": ["papa_bear_(puss_in_boots)"],
        "trigger": ["papa bear \\(puss in boots\\), puss in boots \\(dreamworks\\)"],
    },
    "syrena_(riptideshark)": {
        "character": ["syrena_(riptideshark)"],
        "trigger": ["syrena \\(riptideshark\\), mythology"],
    },
    "princess_fiona": {
        "character": ["princess_fiona"],
        "trigger": ["princess fiona, shrek \\(series\\)"],
    },
    "bunny_raven": {
        "character": ["bunny_raven"],
        "trigger": ["bunny raven, dc comics"],
    },
    "fox_samus_aran": {
        "character": ["fox_samus_aran"],
        "trigger": ["fox samus aran, nintendo"],
    },
    "twitch_(twitch)": {
        "character": ["twitch_(twitch)"],
        "trigger": ["twitch \\(twitch\\), mythology"],
    },
    "larc_(mana)": {
        "character": ["larc_(mana)"],
        "trigger": ["larc \\(mana\\), square enix"],
    },
    "plucky_duck": {
        "character": ["plucky_duck"],
        "trigger": ["plucky duck, warner brothers"],
    },
    "trix_rabbit": {
        "character": ["trix_rabbit"],
        "trigger": ["trix rabbit, general mills"],
    },
    "bender_bending_rodríguez": {
        "character": ["bender_bending_rodríguez"],
        "trigger": ["bender bending rodríguez, comedy central"],
    },
    "cindy_bear": {"character": ["cindy_bear"], "trigger": ["cindy bear, yogi bear"]},
    "fritz_the_cat_(character)": {
        "character": ["fritz_the_cat_(character)"],
        "trigger": ["fritz the cat \\(character\\), fritz the cat"],
    },
    "tetsunoshin": {
        "character": ["tetsunoshin"],
        "trigger": ["tetsunoshin, wan wan celeb soreyuke! tetsunoshin"],
    },
    "sparks_the_raichu": {
        "character": ["sparks_the_raichu"],
        "trigger": ["sparks the raichu, pokemon"],
    },
    "molly_collins": {
        "character": ["molly_collins"],
        "trigger": ["molly collins, cartoon network"],
    },
    "sia_(ebonycrowned)": {
        "character": ["sia_(ebonycrowned)"],
        "trigger": ["sia \\(ebonycrowned\\), mythology"],
    },
    "kiro_(tits)": {
        "character": ["kiro_(tits)"],
        "trigger": ["kiro \\(tits\\), trials in tainted space"],
    },
    "darkwolf_(darkwolfdemon)": {
        "character": ["darkwolf_(darkwolfdemon)"],
        "trigger": ["darkwolf \\(darkwolfdemon\\), egyptian mythology"],
    },
    "ryla": {"character": ["ryla"], "trigger": ["ryla, mythology"]},
    "giru": {"character": ["giru"], "trigger": ["giru, mythology"]},
    "ruben_(djcoyoteguy)": {
        "character": ["ruben_(djcoyoteguy)"],
        "trigger": ["ruben \\(djcoyoteguy\\), bethesda softworks"],
    },
    "electric_spark": {
        "character": ["electric_spark"],
        "trigger": ["electric spark, my little pony"],
    },
    "00284_(character)": {
        "character": ["00284_(character)"],
        "trigger": ["00284 \\(character\\), my little pony"],
    },
    "dragon_princess": {
        "character": ["dragon_princess"],
        "trigger": ["dragon princess, mythology"],
    },
    "fek_(character)": {
        "character": ["fek_(character)"],
        "trigger": ["fek \\(character\\), mythology"],
    },
    "the_imp": {"character": ["the_imp"], "trigger": ["the imp, mythology"]},
    "janja_(the_lion_guard)": {
        "character": ["janja_(the_lion_guard)"],
        "trigger": ["janja \\(the lion guard\\), disney"],
    },
    "sugar_(gats)": {
        "character": ["sugar_(gats)"],
        "trigger": ["sugar \\(gats\\), egyptian mythology"],
    },
    "ginette_cerise_(girokett)": {
        "character": ["ginette_cerise_(girokett)"],
        "trigger": ["ginette cerise \\(girokett\\), halloween"],
    },
    "reimachu": {"character": ["reimachu"], "trigger": ["reimachu, nintendo"]},
    "max_(notkastar)": {
        "character": ["max_(notkastar)"],
        "trigger": ["max \\(notkastar\\), nintendo"],
    },
    "capper_dapperpaws": {
        "character": ["capper_dapperpaws"],
        "trigger": ["capper dapperpaws, my little pony"],
    },
    "kali_(kilinah)": {
        "character": ["kali_(kilinah)"],
        "trigger": ["kali \\(kilinah\\), dustfalconmlp"],
    },
    "dergon_(edjit)": {
        "character": ["dergon_(edjit)"],
        "trigger": ["dergon \\(edjit\\), mythology"],
    },
    "chica_(thevgbear)": {
        "character": ["chica_(thevgbear)"],
        "trigger": ["chica \\(thevgbear\\), scottgames"],
    },
    "moe_(kobold_adventure)": {
        "character": ["moe_(kobold_adventure)"],
        "trigger": ["moe \\(kobold adventure\\), kobold adventure"],
    },
    "gabby_(docbats)": {
        "character": ["gabby_(docbats)"],
        "trigger": ["gabby \\(docbats\\), pokemon"],
    },
    "tank_the_dragon": {
        "character": ["tank_the_dragon"],
        "trigger": ["tank the dragon, mythology"],
    },
    "oata_rinsky": {
        "character": ["oata_rinsky"],
        "trigger": ["oata rinsky, measureup"],
    },
    "shane_gray": {"character": ["shane_gray"], "trigger": ["shane gray, christmas"]},
    "lettuce_(latexia)": {
        "character": ["lettuce_(latexia)"],
        "trigger": ["lettuce \\(latexia\\), nintendo"],
    },
    "lunette_(lunebat)": {
        "character": ["lunette_(lunebat)"],
        "trigger": ["lunette \\(lunebat\\), my little pony"],
    },
    "coach_dale": {"character": ["coach_dale"], "trigger": ["coach dale, halloween"]},
    "andy_renard": {
        "character": ["andy_renard"],
        "trigger": ["andy renard, mythology"],
    },
    "cepheus_(lieutenantskittles)": {
        "character": ["cepheus_(lieutenantskittles)"],
        "trigger": ["cepheus \\(lieutenantskittles\\), mythology"],
    },
    "tiberius_(paladins)": {
        "character": ["tiberius_(paladins)"],
        "trigger": ["tiberius \\(paladins\\), paladins \\(game\\)"],
    },
    "miasma_velenosa_(miasmium)": {
        "character": ["miasma_velenosa_(miasmium)"],
        "trigger": ["miasma velenosa \\(miasmium\\), mythology"],
    },
    "meidri_(interspecies_reviewers)": {
        "character": ["meidri_(interspecies_reviewers)"],
        "trigger": ["meidri \\(interspecies reviewers\\), interspecies reviewers"],
    },
    "bartholomew_martins": {
        "character": ["bartholomew_martins"],
        "trigger": ["bartholomew martins, mythology"],
    },
    "ira_(frozenartifice)": {
        "character": ["ira_(frozenartifice)"],
        "trigger": ["ira \\(frozenartifice\\), mythology"],
    },
    "mocha_latte_love": {
        "character": ["mocha_latte_love"],
        "trigger": ["mocha latte love, mythology"],
    },
    "tao_(gunfire_reborn)": {
        "character": ["tao_(gunfire_reborn)"],
        "trigger": ["tao \\(gunfire reborn\\), gunfire reborn"],
    },
    "mei_marzipan": {
        "character": ["mei_marzipan"],
        "trigger": ["mei marzipan, fuga: melodies of steel"],
    },
    "sasayama_akira": {
        "character": ["sasayama_akira"],
        "trigger": ["sasayama akira, vtuber"],
    },
    "adam_(hazbin_hotel)": {
        "character": ["adam_(hazbin_hotel)"],
        "trigger": ["adam \\(hazbin hotel\\), hazbin hotel"],
    },
    "shaibey_(shaibey)": {
        "character": ["shaibey_(shaibey)"],
        "trigger": ["shaibey \\(shaibey\\), bugsnax"],
    },
    "kittenkeiko": {
        "character": ["kittenkeiko"],
        "trigger": ["kittenkeiko, christmas"],
    },
    "persia_(jay_naylor)": {
        "character": ["persia_(jay_naylor)"],
        "trigger": ["persia \\(jay naylor\\), patreon"],
    },
    "gary_oak": {"character": ["gary_oak"], "trigger": ["gary oak, pokemon"]},
    "kraven_lupei": {
        "character": ["kraven_lupei"],
        "trigger": ["kraven lupei, mythology"],
    },
    "streetdog": {"character": ["streetdog"], "trigger": ["streetdog, disney"]},
    "felinius_cat": {
        "character": ["felinius_cat"],
        "trigger": ["felinius cat, catdog \\(series\\)"],
    },
    "abby_(rukifox)": {
        "character": ["abby_(rukifox)"],
        "trigger": ["abby \\(rukifox\\), halloween"],
    },
    "helen_parr": {"character": ["helen_parr"], "trigger": ["helen parr, disney"]},
    "vault_boy": {"character": ["vault_boy"], "trigger": ["vault boy, fallout"]},
    "maya_(nightfaux)": {
        "character": ["maya_(nightfaux)"],
        "trigger": ["maya \\(nightfaux\\), mythology"],
    },
    "donut_joe_(mlp)": {
        "character": ["donut_joe_(mlp)"],
        "trigger": ["donut joe \\(mlp\\), my little pony"],
    },
    "elbestia_(character)": {
        "character": ["elbestia_(character)"],
        "trigger": ["elbestia \\(character\\), mythology"],
    },
    "fenrir_(amakuchi)": {
        "character": ["fenrir_(amakuchi)"],
        "trigger": ["fenrir \\(amakuchi\\), mythology"],
    },
    "the_laughing_cow": {
        "character": ["the_laughing_cow"],
        "trigger": ["the laughing cow, bel group"],
    },
    "femtoampere_(character)": {
        "character": ["femtoampere_(character)"],
        "trigger": ["femtoampere \\(character\\), mythology"],
    },
    "raven_inkwell_(mlp)": {
        "character": ["raven_inkwell_(mlp)"],
        "trigger": ["raven inkwell \\(mlp\\), my little pony"],
    },
    "grandall_(character)": {
        "character": ["grandall_(character)"],
        "trigger": ["grandall \\(character\\), nintendo"],
    },
    "faxy_(pillo)": {
        "character": ["faxy_(pillo)"],
        "trigger": ["faxy \\(pillo\\), mythology"],
    },
    "nailah": {"character": ["nailah"], "trigger": ["nailah, patreon"]},
    "bri_(ennismore)": {
        "character": ["bri_(ennismore)"],
        "trigger": ["bri \\(ennismore\\), pokemon"],
    },
    "willy_(ohs688)": {
        "character": ["willy_(ohs688)"],
        "trigger": ["willy \\(ohs688\\), mythology"],
    },
    "xefra": {"character": ["xefra"], "trigger": ["xefra, mythology"]},
    "zoey_(dirtyrenamon)": {
        "character": ["zoey_(dirtyrenamon)"],
        "trigger": ["zoey \\(dirtyrenamon\\), dirtyrenamon"],
    },
    "lyon_carter": {"character": ["lyon_carter"], "trigger": ["lyon carter, disney"]},
    "garoh": {"character": ["garoh"], "trigger": ["garoh, mythology"]},
    "elma_(dragon_maid)": {
        "character": ["elma_(dragon_maid)"],
        "trigger": ["elma \\(dragon maid\\), miss kobayashi's dragon maid"],
    },
    "sidni": {"character": ["sidni"], "trigger": ["sidni, dressuptober"]},
    "percey_(character)": {
        "character": ["percey_(character)"],
        "trigger": ["percey \\(character\\), nintendo"],
    },
    "oriana_thaffer": {
        "character": ["oriana_thaffer"],
        "trigger": ["oriana thaffer, disney"],
    },
    "kiko_kempt": {"character": ["kiko_kempt"], "trigger": ["kiko kempt, nintendo"]},
    "gyro_gearloose": {
        "character": ["gyro_gearloose"],
        "trigger": ["gyro gearloose, disney"],
    },
    "judy_reinard": {
        "character": ["judy_reinard"],
        "trigger": ["judy reinard, mythology"],
    },
    "ruin_seeker": {
        "character": ["ruin_seeker"],
        "trigger": ["ruin seeker, tunic \\(video game\\)"],
    },
    "carl_hendricks": {
        "character": ["carl_hendricks"],
        "trigger": ["carl hendricks, echo \\(game\\)"],
    },
    "hanna_fondant": {
        "character": ["hanna_fondant"],
        "trigger": ["hanna fondant, fuga: melodies of steel"],
    },
    "cosmo_(beastars)": {
        "character": ["cosmo_(beastars)"],
        "trigger": ["cosmo \\(beastars\\), beastars"],
    },
    "cev_rosa": {"character": ["cev_rosa"], "trigger": ["cev rosa, nintendo"]},
    "anthony_(goldiescales)": {
        "character": ["anthony_(goldiescales)"],
        "trigger": ["anthony \\(goldiescales\\), mythology"],
    },
    "gunter_(frisky_ferals)": {
        "character": ["gunter_(frisky_ferals)"],
        "trigger": ["gunter \\(frisky ferals\\), frisky ferals"],
    },
    "red_(redeye)": {
        "character": ["red_(redeye)"],
        "trigger": ["red \\(redeye\\), pokemon"],
    },
    "mila_(president_alexander)": {
        "character": ["mila_(president_alexander)"],
        "trigger": ["mila \\(president alexander\\), pokemon"],
    },
    "aira_kokonatsu": {
        "character": ["aira_kokonatsu"],
        "trigger": ["aira kokonatsu, menacing \\(meme\\)"],
    },
    "allie_(rimentus)": {
        "character": ["allie_(rimentus)"],
        "trigger": ["allie \\(rimentus\\), patreon"],
    },
    "circe_(zeptophidia)": {
        "character": ["circe_(zeptophidia)"],
        "trigger": ["circe \\(zeptophidia\\), wizards of the coast"],
    },
    "mara_(wuwutim)": {
        "character": ["mara_(wuwutim)"],
        "trigger": ["mara \\(wuwutim\\), star fox"],
    },
    "dante_(dmc)": {
        "character": ["dante_(dmc)"],
        "trigger": ["dante \\(dmc\\), devil may cry"],
    },
    "tina_lynx": {"character": ["tina_lynx"], "trigger": ["tina lynx, furafterdark"]},
    "weed_(ginga)": {
        "character": ["weed_(ginga)"],
        "trigger": ["weed \\(ginga\\), ginga \\(series\\)"],
    },
    "rita_mordio": {
        "character": ["rita_mordio"],
        "trigger": ["rita mordio, bandai namco"],
    },
    "lulu_(final_fantasy)": {
        "character": ["lulu_(final_fantasy)"],
        "trigger": ["lulu \\(final fantasy\\), square enix"],
    },
    "spike_(extreme_dinosaurs)": {
        "character": ["spike_(extreme_dinosaurs)"],
        "trigger": ["spike \\(extreme dinosaurs\\), extreme dinosaurs"],
    },
    "nesquik_bunny": {
        "character": ["nesquik_bunny"],
        "trigger": ["nesquik bunny, nesquik"],
    },
    "robbie_sinclair": {
        "character": ["robbie_sinclair"],
        "trigger": ["robbie sinclair, dinosaurs \\(series\\)"],
    },
    "inika": {"character": ["inika"], "trigger": ["inika, mythology"]},
    "dixie_(balto)": {
        "character": ["dixie_(balto)"],
        "trigger": ["dixie \\(balto\\), universal studios"],
    },
    "velvet_remedy": {
        "character": ["velvet_remedy"],
        "trigger": ["velvet remedy, my little pony"],
    },
    "katie_tinson": {
        "character": ["katie_tinson"],
        "trigger": ["katie tinson, christmas"],
    },
    "nyx_(mlp)": {
        "character": ["nyx_(mlp)"],
        "trigger": ["nyx \\(mlp\\), my little pony"],
    },
    "midbus": {"character": ["midbus"], "trigger": ["midbus, nintendo"]},
    "dolphin_bomber": {
        "character": ["dolphin_bomber"],
        "trigger": ["dolphin bomber, bomberman jetters"],
    },
    "ice_queen": {
        "character": ["ice_queen"],
        "trigger": ["ice queen, cartoon network"],
    },
    "scp-173": {"character": ["scp-173"], "trigger": ["scp-173, scp foundation"]},
    "dreamsicle_swirl": {
        "character": ["dreamsicle_swirl"],
        "trigger": ["dreamsicle swirl, my little pony"],
    },
    "cross-fox": {"character": ["cross-fox"], "trigger": ["cross-fox, disney"]},
    "harbinger_the_outworld_devourer": {
        "character": ["harbinger_the_outworld_devourer"],
        "trigger": ["harbinger the outworld devourer, dota"],
    },
    "balanar_the_night_stalker": {
        "character": ["balanar_the_night_stalker"],
        "trigger": ["balanar the night stalker, dota"],
    },
    "emma_the_eevee": {
        "character": ["emma_the_eevee"],
        "trigger": ["emma the eevee, pokemon"],
    },
    "rolf_(animal_crossing)": {
        "character": ["rolf_(animal_crossing)"],
        "trigger": ["rolf \\(animal crossing\\), animal crossing"],
    },
    "nova_(warframe)": {
        "character": ["nova_(warframe)"],
        "trigger": ["nova \\(warframe\\), warframe"],
    },
    "zack_fox": {"character": ["zack_fox"], "trigger": ["zack fox, mythology"]},
    "jay_(sammfeatblueheart)": {
        "character": ["jay_(sammfeatblueheart)"],
        "trigger": ["jay \\(sammfeatblueheart\\), city feathers"],
    },
    "opal_(animal_crossing)": {
        "character": ["opal_(animal_crossing)"],
        "trigger": ["opal \\(animal crossing\\), animal crossing"],
    },
    "giovanni_da_milano": {
        "character": ["giovanni_da_milano"],
        "trigger": ["giovanni da milano, starbound"],
    },
    "howlfei": {"character": ["howlfei"], "trigger": ["howlfei, mythology"]},
    "gariyuu": {"character": ["gariyuu"], "trigger": ["gariyuu, mythology"]},
    "greater_dog": {
        "character": ["greater_dog"],
        "trigger": ["greater dog, undertale \\(series\\)"],
    },
    "endogeny": {
        "character": ["endogeny"],
        "trigger": ["endogeny, undertale \\(series\\)"],
    },
    "sans_(underfell)": {
        "character": ["sans_(underfell)"],
        "trigger": ["sans \\(underfell\\), undertale \\(series\\)"],
    },
    "ivara_(warframe)": {
        "character": ["ivara_(warframe)"],
        "trigger": ["ivara \\(warframe\\), warframe"],
    },
    "reik_(peritian)": {
        "character": ["reik_(peritian)"],
        "trigger": ["reik \\(peritian\\), adam lambert"],
    },
    "samantha_thott": {
        "character": ["samantha_thott"],
        "trigger": ["samantha thott, doom eternal"],
    },
    "knipp_(knipp)": {
        "character": ["knipp_(knipp)"],
        "trigger": ["knipp \\(knipp\\), mythology"],
    },
    "icarus_aresane": {
        "character": ["icarus_aresane"],
        "trigger": ["icarus aresane, mythology"],
    },
    "tristan_knight": {
        "character": ["tristan_knight"],
        "trigger": ["tristan knight, shirt cut meme"],
    },
    "hudson_(zp92)": {
        "character": ["hudson_(zp92)"],
        "trigger": ["hudson \\(zp92\\), mythology"],
    },
    "kenta_yamashita": {
        "character": ["kenta_yamashita"],
        "trigger": ["kenta yamashita, texnatsu"],
    },
    "vex_(alibiwolf)": {
        "character": ["vex_(alibiwolf)"],
        "trigger": ["vex \\(alibiwolf\\), mythology"],
    },
    "delirium_(tboi)": {
        "character": ["delirium_(tboi)"],
        "trigger": ["delirium \\(tboi\\), the binding of isaac \\(series\\)"],
    },
    "vic_(animal_crossing)": {
        "character": ["vic_(animal_crossing)"],
        "trigger": ["vic \\(animal crossing\\), animal crossing"],
    },
    "tanngrisnir_(tas)": {
        "character": ["tanngrisnir_(tas)"],
        "trigger": ["tanngrisnir \\(tas\\), lifewonders"],
    },
    "kurly": {"character": ["kurly"], "trigger": ["kurly, pokemon"]},
    "falcon_woodwere": {
        "character": ["falcon_woodwere"],
        "trigger": ["falcon woodwere, titanfall 2"],
    },
    "freckles_(nukepone)": {
        "character": ["freckles_(nukepone)"],
        "trigger": ["freckles \\(nukepone\\), mythology"],
    },
    "marcus_(thehades)": {
        "character": ["marcus_(thehades)"],
        "trigger": ["marcus \\(thehades\\), mythology"],
    },
    "monique_(atrolux)": {
        "character": ["monique_(atrolux)"],
        "trigger": ["monique \\(atrolux\\), jenga"],
    },
    "jay_(confusedraven)": {
        "character": ["jay_(confusedraven)"],
        "trigger": ["jay \\(confusedraven\\), piper perri surrounded"],
    },
    "cerberus_the_demogorgon": {
        "character": ["cerberus_the_demogorgon"],
        "trigger": ["cerberus the demogorgon, stranger things"],
    },
    "aeiou_(yoako)": {
        "character": ["aeiou_(yoako)"],
        "trigger": ["aeiou \\(yoako\\), meme clothing"],
    },
    "mina_(mina_the_hollower)": {
        "character": ["mina_(mina_the_hollower)"],
        "trigger": ["mina \\(mina the hollower\\), yacht club games"],
    },
    "queen_octavia_(teathekook)": {
        "character": ["queen_octavia_(teathekook)"],
        "trigger": ["queen octavia \\(teathekook\\), helluva boss"],
    },
    "ishmael_(core34)": {
        "character": ["ishmael_(core34)"],
        "trigger": ["ishmael \\(core34\\), mythology"],
    },
    "goobie_(da3rd)": {
        "character": ["goobie_(da3rd)"],
        "trigger": ["goobie \\(da3rd\\), mythology"],
    },
    "aym_(cult_of_the_lamb)": {
        "character": ["aym_(cult_of_the_lamb)"],
        "trigger": ["aym \\(cult of the lamb\\), cult of the lamb"],
    },
    "moxy_(grimart)": {
        "character": ["moxy_(grimart)"],
        "trigger": ["moxy \\(grimart\\), dodge \\(brand\\)"],
    },
    "adam_(juicyducksfm)": {
        "character": ["adam_(juicyducksfm)"],
        "trigger": ["adam \\(juicyducksfm\\), sonic the hedgehog \\(series\\)"],
    },
    "gordon_freeman": {
        "character": ["gordon_freeman"],
        "trigger": ["gordon freeman, valve"],
    },
    "sabuteur": {"character": ["sabuteur"], "trigger": ["sabuteur, mythology"]},
    "new_brian": {"character": ["new_brian"], "trigger": ["new brian, family guy"]},
    "mayhem_(renard)": {
        "character": ["mayhem_(renard)"],
        "trigger": ["mayhem \\(renard\\), lapfox trax"],
    },
    "dora_marquez": {
        "character": ["dora_marquez"],
        "trigger": ["dora marquez, dora the explorer"],
    },
    "trassk": {"character": ["trassk"], "trigger": ["trassk, greek mythology"]},
    "dale_(furronika)": {
        "character": ["dale_(furronika)"],
        "trigger": ["dale \\(furronika\\), disney"],
    },
    "foxydude": {"character": ["foxydude"], "trigger": ["foxydude, nintendo"]},
    "hellboy_(character)": {
        "character": ["hellboy_(character)"],
        "trigger": ["hellboy \\(character\\), hellboy \\(series\\)"],
    },
    "omega": {"character": ["omega"], "trigger": ["omega, mythology"]},
    "wyla_(character)": {
        "character": ["wyla_(character)"],
        "trigger": ["wyla \\(character\\), mythology"],
    },
    "pumzie": {"character": ["pumzie"], "trigger": ["pumzie, mythology"]},
    "leo_(thetwfz)": {
        "character": ["leo_(thetwfz)"],
        "trigger": ["leo \\(thetwfz\\), pokemon"],
    },
    "roy_(raichu)": {
        "character": ["roy_(raichu)"],
        "trigger": ["roy \\(raichu\\), pokemon"],
    },
    "gavin_(tokifuji)": {
        "character": ["gavin_(tokifuji)"],
        "trigger": ["gavin \\(tokifuji\\), patreon"],
    },
    "zoqi": {"character": ["zoqi"], "trigger": ["zoqi, halloween"]},
    "varanis_blackclaw": {
        "character": ["varanis_blackclaw"],
        "trigger": ["varanis blackclaw, mythology"],
    },
    "squirrelflight_(warriors)": {
        "character": ["squirrelflight_(warriors)"],
        "trigger": ["squirrelflight \\(warriors\\), warriors \\(book series\\)"],
    },
    "pikachu_pop_star": {
        "character": ["pikachu_pop_star"],
        "trigger": ["pikachu pop star, pokemon"],
    },
    "sundyz": {"character": ["sundyz"], "trigger": ["sundyz, mythology"]},
    "komane": {"character": ["komane"], "trigger": ["komane, yo-kai watch"]},
    "mew_mew_(undertale)": {
        "character": ["mew_mew_(undertale)"],
        "trigger": ["mew mew \\(undertale\\), undertale \\(series\\)"],
    },
    "shelby_(simplifypm)": {
        "character": ["shelby_(simplifypm)"],
        "trigger": ["shelby \\(simplifypm\\), pokemon"],
    },
    "eldrick_pica": {
        "character": ["eldrick_pica"],
        "trigger": ["eldrick pica, nintendo"],
    },
    "rela": {"character": ["rela"], "trigger": ["rela, patreon"]},
    "ria_(gnoll)": {
        "character": ["ria_(gnoll)"],
        "trigger": ["ria \\(gnoll\\), mythology"],
    },
    "kay_(thiccvally)": {
        "character": ["kay_(thiccvally)"],
        "trigger": ["kay \\(thiccvally\\), nintendo"],
    },
    "verlo_streams": {
        "character": ["verlo_streams"],
        "trigger": ["verlo streams, mythology"],
    },
    "kronas": {"character": ["kronas"], "trigger": ["kronas, mythology"]},
    "bfw": {"character": ["bfw"], "trigger": ["bfw, mythology"]},
    "lenni_(artlegionary)": {
        "character": ["lenni_(artlegionary)"],
        "trigger": ["lenni \\(artlegionary\\), christmas"],
    },
    "beau_(luxurias)": {
        "character": ["beau_(luxurias)"],
        "trigger": ["beau \\(luxurias\\), christmas"],
    },
    "fynnley_(character)": {
        "character": ["fynnley_(character)"],
        "trigger": ["fynnley \\(character\\), pokemon"],
    },
    "styrling": {"character": ["styrling"], "trigger": ["styrling, mythology"]},
    "argo_northrop": {
        "character": ["argo_northrop"],
        "trigger": ["argo northrop, knights college"],
    },
    "damien_woof": {
        "character": ["damien_woof"],
        "trigger": ["damien woof, mythology"],
    },
    "alberto_scorfano": {
        "character": ["alberto_scorfano"],
        "trigger": ["alberto scorfano, disney"],
    },
    "nooshy_(sing)": {
        "character": ["nooshy_(sing)"],
        "trigger": ["nooshy \\(sing\\), illumination entertainment"],
    },
    "jordyn_(goobysart)": {
        "character": ["jordyn_(goobysart)"],
        "trigger": ["jordyn \\(goobysart\\), sony corporation"],
    },
    "kyle_kendricks_(forestdale)": {
        "character": ["kyle_kendricks_(forestdale)"],
        "trigger": ["kyle kendricks \\(forestdale\\), forestdale"],
    },
    "yac_(jigglyjuggle)": {
        "character": ["yac_(jigglyjuggle)"],
        "trigger": ["yac \\(jigglyjuggle\\)"],
    },
    "warfare_gardevoir": {
        "character": ["warfare_gardevoir"],
        "trigger": ["warfare gardevoir, pokemon"],
    },
    "oscar_(mgferret)": {
        "character": ["oscar_(mgferret)"],
        "trigger": ["oscar \\(mgferret\\)"],
    },
    "bullfrog_(captain_laserhawk)": {
        "character": ["bullfrog_(captain_laserhawk)"],
        "trigger": [
            "bullfrog \\(captain laserhawk\\), captain laserhawk: a blood dragon remix"
        ],
    },
    "void": {"character": ["void"], "trigger": ["void, riot games"]},
    "sukebe": {"character": ["sukebe"], "trigger": ["sukebe, mythology"]},
    "marty_(onta)": {
        "character": ["marty_(onta)"],
        "trigger": ["marty \\(onta\\), hardblush"],
    },
    "keller_(kellervo)": {
        "character": ["keller_(kellervo)"],
        "trigger": ["keller \\(kellervo\\), mythology"],
    },
    "oolong_(dragon_ball)": {
        "character": ["oolong_(dragon_ball)"],
        "trigger": ["oolong \\(dragon ball\\), dragon ball"],
    },
    "jazz_jackrabbit": {
        "character": ["jazz_jackrabbit"],
        "trigger": ["jazz jackrabbit, epic games"],
    },
    "neytiri": {
        "character": ["neytiri"],
        "trigger": ["neytiri, james cameron's avatar"],
    },
    "claire_o'conell": {
        "character": ["claire_o'conell"],
        "trigger": ["claire o'conell, uberquest"],
    },
    "dr._hutchison": {
        "character": ["dr._hutchison"],
        "trigger": ["dr. hutchison, rocko's modern life"],
    },
    "pepper_ackerman": {
        "character": ["pepper_ackerman"],
        "trigger": ["pepper ackerman, mythology"],
    },
    "chris_(meesh)": {
        "character": ["chris_(meesh)"],
        "trigger": ["chris \\(meesh\\), little buddy"],
    },
    "mamizou_futatsuiwa": {
        "character": ["mamizou_futatsuiwa"],
        "trigger": ["mamizou futatsuiwa, touhou"],
    },
    "cinnamon_swirl": {
        "character": ["cinnamon_swirl"],
        "trigger": ["cinnamon swirl, pokemon"],
    },
    "grizzly_(shirokuma_cafe)": {
        "character": ["grizzly_(shirokuma_cafe)"],
        "trigger": ["grizzly \\(shirokuma cafe\\), shirokuma cafe"],
    },
    "hondo_flanks_(mlp)": {
        "character": ["hondo_flanks_(mlp)"],
        "trigger": ["hondo flanks \\(mlp\\), my little pony"],
    },
    "shino_asada": {
        "character": ["shino_asada"],
        "trigger": ["shino asada, sword art online"],
    },
    "fuchsia_(animal_crossing)": {
        "character": ["fuchsia_(animal_crossing)"],
        "trigger": ["fuchsia \\(animal crossing\\), animal crossing"],
    },
    "masked_matter-horn_(mlp)": {
        "character": ["masked_matter-horn_(mlp)"],
        "trigger": ["masked matter-horn \\(mlp\\), my little pony"],
    },
    "azul_draconis": {
        "character": ["azul_draconis"],
        "trigger": ["azul draconis, mythology"],
    },
    "wolfyhero": {"character": ["wolfyhero"], "trigger": ["wolfyhero, pokemon"]},
    "laito": {"character": ["laito"], "trigger": ["laito, mythology"]},
    "ken_(garouzuki)": {
        "character": ["ken_(garouzuki)"],
        "trigger": ["ken \\(garouzuki\\), christmas"],
    },
    "dianna_(komponi)": {
        "character": ["dianna_(komponi)"],
        "trigger": ["dianna \\(komponi\\), mythology"],
    },
    "corrin_(fire_emblem)": {
        "character": ["corrin_(fire_emblem)"],
        "trigger": ["corrin \\(fire emblem\\), fire emblem"],
    },
    "tatsumaki": {"character": ["tatsumaki"], "trigger": ["tatsumaki, one-punch man"]},
    "tristen": {"character": ["tristen"], "trigger": ["tristen, the elder scrolls"]},
    "kordi": {"character": ["kordi"], "trigger": ["kordi, christmas"]},
    "reggie_(james_howard)": {
        "character": ["reggie_(james_howard)"],
        "trigger": ["reggie \\(james howard\\), patreon"],
    },
    "beverly_(athiesh)": {
        "character": ["beverly_(athiesh)"],
        "trigger": ["beverly \\(athiesh\\), riot games"],
    },
    "sona_(noxiis)": {
        "character": ["sona_(noxiis)"],
        "trigger": ["sona \\(noxiis\\), mythology"],
    },
    "asterius_(tas)": {
        "character": ["asterius_(tas)"],
        "trigger": ["asterius \\(tas\\), lifewonders"],
    },
    "pinkfong_(character)": {
        "character": ["pinkfong_(character)"],
        "trigger": ["pinkfong \\(character\\), baby shark"],
    },
    "lefty_(fnaf)": {
        "character": ["lefty_(fnaf)"],
        "trigger": ["lefty \\(fnaf\\), scottgames"],
    },
    "tao_(rubber)": {
        "character": ["tao_(rubber)"],
        "trigger": ["tao \\(rubber\\), dreamworks"],
    },
    "gyobu's_underlings": {
        "character": ["gyobu's_underlings"],
        "trigger": ["gyobu's underlings, lifewonders"],
    },
    "avis_(matchaghost)": {
        "character": ["avis_(matchaghost)"],
        "trigger": ["avis \\(matchaghost\\), pokemon"],
    },
    "elliot_(unpopularwolf)": {
        "character": ["elliot_(unpopularwolf)"],
        "trigger": ["elliot \\(unpopularwolf\\), mythology"],
    },
    "soarin_(soarinarts)": {
        "character": ["soarin_(soarinarts)"],
        "trigger": ["soarin \\(soarinarts\\), mythology"],
    },
    "rose_(limebreaker)": {
        "character": ["rose_(limebreaker)"],
        "trigger": ["rose \\(limebreaker\\), mythology"],
    },
    "ranger_rabbit": {
        "character": ["ranger_rabbit"],
        "trigger": ["ranger rabbit, elinor wonders why"],
    },
    "aimi_(sleepysushiroll)": {
        "character": ["aimi_(sleepysushiroll)"],
        "trigger": ["aimi \\(sleepysushiroll\\), undertale \\(series\\)"],
    },
    "clara_(canisfidelis)": {
        "character": ["clara_(canisfidelis)"],
        "trigger": ["clara \\(canisfidelis\\), no nut november"],
    },
    "stella_(gvh)": {
        "character": ["stella_(gvh)"],
        "trigger": ["stella \\(gvh\\), goodbye volcano high"],
    },
    "torchy": {"character": ["torchy"], "trigger": ["torchy, pokemon"]},
    "yae_miko": {"character": ["yae_miko"], "trigger": ["yae miko, mihoyo"]},
    "darren_(martincorps)": {
        "character": ["darren_(martincorps)"],
        "trigger": ["darren \\(martincorps\\), chinese zodiac"],
    },
    "videlthewusky": {
        "character": ["videlthewusky"],
        "trigger": ["videlthewusky, mythology"],
    },
    "tarahe": {"character": ["tarahe"], "trigger": ["tarahe, mythology"]},
    "aurora_(nbanoob)": {
        "character": ["aurora_(nbanoob)"],
        "trigger": ["aurora \\(nbanoob\\), pokemon"],
    },
    "cherri_topps": {
        "character": ["cherri_topps"],
        "trigger": ["cherri topps, jurassic beauties"],
    },
    "albert_lupine": {
        "character": ["albert_lupine"],
        "trigger": ["albert lupine, nintendo"],
    },
    "wildmutt": {"character": ["wildmutt"], "trigger": ["wildmutt, cartoon network"]},
    "chewbacca": {"character": ["chewbacca"], "trigger": ["chewbacca, star wars"]},
    "ruby_(animal_crossing)": {
        "character": ["ruby_(animal_crossing)"],
        "trigger": ["ruby \\(animal crossing\\), animal crossing"],
    },
    "porky_minch": {
        "character": ["porky_minch"],
        "trigger": ["porky minch, earthbound \\(series\\)"],
    },
    "murray_hippopotamus": {
        "character": ["murray_hippopotamus"],
        "trigger": ["murray hippopotamus, sucker punch productions"],
    },
    "lucky_wolf": {"character": ["lucky_wolf"], "trigger": ["lucky wolf, mythology"]},
    "meg_griffin": {
        "character": ["meg_griffin"],
        "trigger": ["meg griffin, family guy"],
    },
    "molly_macdonald": {
        "character": ["molly_macdonald"],
        "trigger": ["molly macdonald, arthur \\(series\\)"],
    },
    "skipper_(madagascar)": {
        "character": ["skipper_(madagascar)"],
        "trigger": ["skipper \\(madagascar\\), dreamworks"],
    },
    "varby": {"character": ["varby"], "trigger": ["varby, mythology"]},
    "master_mantis": {
        "character": ["master_mantis"],
        "trigger": ["master mantis, kung fu panda"],
    },
    "big_brian": {"character": ["big_brian"], "trigger": ["big brian, my little pony"]},
    "xeila": {"character": ["xeila"], "trigger": ["xeila, mythology"]},
    "queen_oriana": {
        "character": ["queen_oriana"],
        "trigger": ["queen oriana, el arca"],
    },
    "talking_angela": {
        "character": ["talking_angela"],
        "trigger": ["talking angela, talking tom and friends"],
    },
    "nyx_(warframe)": {
        "character": ["nyx_(warframe)"],
        "trigger": ["nyx \\(warframe\\), warframe"],
    },
    "barbera_(regular_show)": {
        "character": ["barbera_(regular_show)"],
        "trigger": ["barbera \\(regular show\\), cartoon network"],
    },
    "cuddles_(character)": {
        "character": ["cuddles_(character)"],
        "trigger": ["cuddles \\(character\\), pokemon"],
    },
    "dominique_(bionichound)": {
        "character": ["dominique_(bionichound)"],
        "trigger": ["dominique \\(bionichound\\), mythology"],
    },
    "pharah_(overwatch)": {
        "character": ["pharah_(overwatch)"],
        "trigger": ["pharah \\(overwatch\\), overwatch"],
    },
    "cole_cassidy": {
        "character": ["cole_cassidy"],
        "trigger": ["cole cassidy, blizzard entertainment"],
    },
    "brutus_(twokinds)": {
        "character": ["brutus_(twokinds)"],
        "trigger": ["brutus \\(twokinds\\), twokinds"],
    },
    "mod_(glacierclear)": {
        "character": ["mod_(glacierclear)"],
        "trigger": ["mod \\(glacierclear\\), pokemon"],
    },
    "pier_(felino)": {
        "character": ["pier_(felino)"],
        "trigger": ["pier \\(felino\\), nintendo"],
    },
    "evan_(kihu)": {
        "character": ["evan_(kihu)"],
        "trigger": ["evan \\(kihu\\), mythology"],
    },
    "saber_(firestorm3)": {
        "character": ["saber_(firestorm3)"],
        "trigger": ["saber \\(firestorm3\\), eklund daily life in a royal family"],
    },
    "llori_gray": {"character": ["llori_gray"], "trigger": ["llori gray, skype"]},
    "wildfire_(rubberbuns)": {
        "character": ["wildfire_(rubberbuns)"],
        "trigger": ["wildfire \\(rubberbuns\\), super smash bros."],
    },
    "momo_(creepypasta)": {
        "character": ["momo_(creepypasta)"],
        "trigger": ["momo \\(creepypasta\\), creepypasta"],
    },
    "lover_(coldfrontvelvet)": {
        "character": ["lover_(coldfrontvelvet)"],
        "trigger": ["lover \\(coldfrontvelvet\\), nintendo"],
    },
    "ember_nifflehiem": {
        "character": ["ember_nifflehiem"],
        "trigger": ["ember nifflehiem, mythology"],
    },
    "lira_(joaoppereiraus)": {
        "character": ["lira_(joaoppereiraus)"],
        "trigger": ["lira \\(joaoppereiraus\\), tales of sezvilpan"],
    },
    "tauski": {"character": ["tauski"], "trigger": ["tauski, nintendo switch"]},
    "alduin_hearth_(character)": {
        "character": ["alduin_hearth_(character)"],
        "trigger": ["alduin hearth \\(character\\), hazbin hotel"],
    },
    "marie_(oughta)": {
        "character": ["marie_(oughta)"],
        "trigger": ["marie \\(oughta\\), christmas"],
    },
    "allandi": {"character": ["allandi"], "trigger": ["allandi, pokemon"]},
    "loona_(carbiid3)": {
        "character": ["loona_(carbiid3)"],
        "trigger": ["loona \\(carbiid3\\), helluva boss"],
    },
    "jenjen_(oyenvar)": {
        "character": ["jenjen_(oyenvar)"],
        "trigger": ["jenjen \\(oyenvar\\), meme clothing"],
    },
    "bibbo_(oc)": {
        "character": ["bibbo_(oc)"],
        "trigger": ["bibbo \\(oc\\), my little pony"],
    },
    "justice_(helltaker)": {
        "character": ["justice_(helltaker)"],
        "trigger": ["justice \\(helltaker\\), helltaker"],
    },
    "tote_brando": {
        "character": ["tote_brando"],
        "trigger": ["tote brando, made in abyss"],
    },
    "ondrea_(ondrea)": {
        "character": ["ondrea_(ondrea)"],
        "trigger": ["ondrea \\(ondrea\\), mythology"],
    },
    "nolani_(quin-nsfw)": {
        "character": ["nolani_(quin-nsfw)"],
        "trigger": ["nolani \\(quin-nsfw\\), greek mythology"],
    },
    "itzamna_(tas)": {
        "character": ["itzamna_(tas)"],
        "trigger": ["itzamna \\(tas\\), lifewonders"],
    },
    "tuntematon": {"character": ["tuntematon"], "trigger": ["tuntematon, mythology"]},
    "gabu_(crave_saga)": {
        "character": ["gabu_(crave_saga)"],
        "trigger": ["gabu \\(crave saga\\), crave saga"],
    },
    "light_dragon_(totk)": {
        "character": ["light_dragon_(totk)"],
        "trigger": ["light dragon \\(totk\\), the legend of zelda"],
    },
    "chaco_(cave_story)": {
        "character": ["chaco_(cave_story)"],
        "trigger": ["chaco \\(cave story\\), cave story"],
    },
    "sephiroth_(final_fantasy_vii)": {
        "character": ["sephiroth_(final_fantasy_vii)"],
        "trigger": ["sephiroth \\(final fantasy vii\\), final fantasy vii"],
    },
    "roxikat": {"character": ["roxikat"], "trigger": ["roxikat, supermegatopia"]},
    "ropes_(character)": {
        "character": ["ropes_(character)"],
        "trigger": ["ropes \\(character\\), mythology"],
    },
    "matilda_(adventures_in_bushtown)": {
        "character": ["matilda_(adventures_in_bushtown)"],
        "trigger": [
            "matilda \\(adventures in bushtown\\), skippy: adventures in bushtown"
        ],
    },
    "tsume_zyzco": {
        "character": ["tsume_zyzco"],
        "trigger": ["tsume zyzco, mythology"],
    },
    "frosti_loxxxe": {
        "character": ["frosti_loxxxe"],
        "trigger": ["frosti loxxxe, sonic the hedgehog \\(series\\)"],
    },
    "smokey_bear": {
        "character": ["smokey_bear"],
        "trigger": ["smokey bear, united states forest service"],
    },
    "charles_entertainment_cheese": {
        "character": ["charles_entertainment_cheese"],
        "trigger": ["charles entertainment cheese, chuck e. cheese's pizzeria"],
    },
    "power_girl": {"character": ["power_girl"], "trigger": ["power girl, dc comics"]},
    "maximus_(tangled)": {
        "character": ["maximus_(tangled)"],
        "trigger": ["maximus \\(tangled\\), disney's tangled \\(film\\)"],
    },
    "princess_shroob": {
        "character": ["princess_shroob"],
        "trigger": ["princess shroob, mario bros"],
    },
    "whip_(dreamkeepers)": {
        "character": ["whip_(dreamkeepers)"],
        "trigger": ["whip \\(dreamkeepers\\), dreamkeepers"],
    },
    "boris_(theboris)": {
        "character": ["boris_(theboris)"],
        "trigger": ["boris \\(theboris\\), mythology"],
    },
    "mako_(rudragon)": {
        "character": ["mako_(rudragon)"],
        "trigger": ["mako \\(rudragon\\), mythology"],
    },
    "oikawa_shizuku": {
        "character": ["oikawa_shizuku"],
        "trigger": ["oikawa shizuku, cygames"],
    },
    "amy_rose_the_werehog": {
        "character": ["amy_rose_the_werehog"],
        "trigger": ["amy rose the werehog, sonic the hedgehog \\(series\\)"],
    },
    "poojawa": {"character": ["poojawa"], "trigger": ["poojawa, mythology"]},
    "catherine_applebottom": {
        "character": ["catherine_applebottom"],
        "trigger": ["catherine applebottom, hollandworks"],
    },
    "frances_sugarfoot": {
        "character": ["frances_sugarfoot"],
        "trigger": ["frances sugarfoot, halloween"],
    },
    "beanie_(roommates)": {
        "character": ["beanie_(roommates)"],
        "trigger": ["beanie \\(roommates\\), roommates:motha"],
    },
    "batt_the_bat": {
        "character": ["batt_the_bat"],
        "trigger": ["batt the bat, google"],
    },
    "exeter": {"character": ["exeter"], "trigger": ["exeter, mythology"]},
    "steelfire": {"character": ["steelfire"], "trigger": ["steelfire, mythology"]},
    "tarah_(fvt)": {
        "character": ["tarah_(fvt)"],
        "trigger": ["tarah \\(fvt\\), fairies vs tentacles"],
    },
    "selix": {"character": ["selix"], "trigger": ["selix, nintendo"]},
    "lillianwinters": {
        "character": ["lillianwinters"],
        "trigger": ["lillianwinters, mythology"],
    },
    "steel_cat_(character)": {
        "character": ["steel_cat_(character)"],
        "trigger": ["steel cat \\(character\\), pokemon"],
    },
    "tutori": {"character": ["tutori"], "trigger": ["tutori, undertale \\(series\\)"]},
    "li_shan_(kung_fu_panda)": {
        "character": ["li_shan_(kung_fu_panda)"],
        "trigger": ["li shan \\(kung fu panda\\), kung fu panda"],
    },
    "milo_(captain_nikko)": {
        "character": ["milo_(captain_nikko)"],
        "trigger": ["milo \\(captain nikko\\), patreon"],
    },
    "abebi_(zp92)": {
        "character": ["abebi_(zp92)"],
        "trigger": ["abebi \\(zp92\\), tribez"],
    },
    "aunt_molly_(nitw)": {
        "character": ["aunt_molly_(nitw)"],
        "trigger": ["aunt molly \\(nitw\\), night in the woods"],
    },
    "benji_(mainlion)": {
        "character": ["benji_(mainlion)"],
        "trigger": ["benji \\(mainlion\\), apple inc."],
    },
    "ragnacock": {
        "character": ["ragnacock"],
        "trigger": ["ragnacock, sonic the hedgehog \\(series\\)"],
    },
    "tumble_the_skunk": {
        "character": ["tumble_the_skunk"],
        "trigger": ["tumble the skunk, sonic the hedgehog \\(series\\)"],
    },
    "azukipuddles": {
        "character": ["azukipuddles"],
        "trigger": ["azukipuddles, mythology"],
    },
    "dorian_(bds_charmeleon)": {
        "character": ["dorian_(bds_charmeleon)"],
        "trigger": ["dorian \\(bds charmeleon\\), pokemon"],
    },
    "gallgard": {"character": ["gallgard"], "trigger": ["gallgard, patreon"]},
    "shannon_shark": {
        "character": ["shannon_shark"],
        "trigger": ["shannon shark, i mean breast milk"],
    },
    "puffchu": {"character": ["puffchu"], "trigger": ["puffchu, pokemon"]},
    "kinako_(40hara)": {
        "character": ["kinako_(40hara)"],
        "trigger": ["kinako \\(40hara\\), new year"],
    },
    "brulee_(y11)": {
        "character": ["brulee_(y11)"],
        "trigger": ["brulee \\(y11\\), mythology"],
    },
    "cheems": {"character": ["cheems"], "trigger": ["cheems, dogelore"]},
    "tonitrux": {"character": ["tonitrux"], "trigger": ["tonitrux, mythology"]},
    "reku_(akubon)": {
        "character": ["reku_(akubon)"],
        "trigger": ["reku \\(akubon\\), nintendo"],
    },
    "kotaro_(leobo)": {
        "character": ["kotaro_(leobo)"],
        "trigger": ["kotaro \\(leobo\\), mythology"],
    },
    "kanon_(applejacksville)": {
        "character": ["kanon_(applejacksville)"],
        "trigger": ["kanon \\(applejacksville\\), mythology"],
    },
    "kay_(1-upclock)": {
        "character": ["kay_(1-upclock)"],
        "trigger": ["kay \\(1-upclock\\), pokemon"],
    },
    "selune_darkeye": {
        "character": ["selune_darkeye"],
        "trigger": ["selune darkeye, my little pony"],
    },
    "rabbid_rosalina": {
        "character": ["rabbid_rosalina"],
        "trigger": ["rabbid rosalina, raving rabbids"],
    },
    "maple_(maplegek)": {
        "character": ["maple_(maplegek)"],
        "trigger": ["maple \\(maplegek\\), mythology"],
    },
    "sally_foxheart": {
        "character": ["sally_foxheart"],
        "trigger": ["sally foxheart, sailor moon \\(series\\)"],
    },
    "akeno_(itsnafulol)": {
        "character": ["akeno_(itsnafulol)"],
        "trigger": ["akeno \\(itsnafulol\\), pokemon"],
    },
    "vaporunny_(follygee)": {
        "character": ["vaporunny_(follygee)"],
        "trigger": ["vaporunny \\(follygee\\), pokemon"],
    },
    "reina_(vinqou)": {
        "character": ["reina_(vinqou)"],
        "trigger": ["reina \\(vinqou\\), caramelldansen"],
    },
    "suvi_(elronya)": {
        "character": ["suvi_(elronya)"],
        "trigger": ["suvi \\(elronya\\), halloween"],
    },
    "squeek": {"character": ["squeek"], "trigger": ["squeek, clubstripes"]},
    "professor_oak": {
        "character": ["professor_oak"],
        "trigger": ["professor oak, pokemon"],
    },
    "dozer_(braford)": {
        "character": ["dozer_(braford)"],
        "trigger": ["dozer \\(braford\\), house of beef"],
    },
    "nikki_blackcat": {
        "character": ["nikki_blackcat"],
        "trigger": ["nikki blackcat, kama sutra"],
    },
    "myrl": {"character": ["myrl"], "trigger": ["myrl, mythology"]},
    "goji_(flitchee)": {
        "character": ["goji_(flitchee)"],
        "trigger": ["goji \\(flitchee\\), mythology"],
    },
    "tiamat_(god)": {
        "character": ["tiamat_(god)"],
        "trigger": ["tiamat \\(god\\), mythology"],
    },
    "jad'thor": {"character": ["jad'thor"], "trigger": ["jad'thor, mitsubishi"]},
    "shenron": {"character": ["shenron"], "trigger": ["shenron, mythology"]},
    "alduin": {"character": ["alduin"], "trigger": ["alduin, the elder scrolls"]},
    "bandit_(holidaypup)": {
        "character": ["bandit_(holidaypup)"],
        "trigger": ["bandit \\(holidaypup\\), disney"],
    },
    "kha'zix_(lol)": {
        "character": ["kha'zix_(lol)"],
        "trigger": ["kha'zix \\(lol\\), riot games"],
    },
    "john_carter": {
        "character": ["john_carter"],
        "trigger": ["john carter, a princess of mars"],
    },
    "ernesto_(rebeldragon101)": {
        "character": ["ernesto_(rebeldragon101)"],
        "trigger": ["ernesto \\(rebeldragon101\\), mythology"],
    },
    "doggydog_(character)": {
        "character": ["doggydog_(character)"],
        "trigger": ["doggydog \\(character\\), furaffinity"],
    },
    "midori_gel": {
        "character": ["midori_gel"],
        "trigger": ["midori gel, my little pony"],
    },
    "penny_carson": {
        "character": ["penny_carson"],
        "trigger": ["penny carson, netflix"],
    },
    "alexandra_(david_siegl)": {
        "character": ["alexandra_(david_siegl)"],
        "trigger": ["alexandra \\(david siegl\\)"],
    },
    "mairi_nigalya_ponya": {
        "character": ["mairi_nigalya_ponya"],
        "trigger": ["mairi nigalya ponya, nintendo"],
    },
    "dannydumal": {"character": ["dannydumal"], "trigger": ["dannydumal, mestiso"]},
    "big_booty_pikachu": {
        "character": ["big_booty_pikachu"],
        "trigger": ["big booty pikachu, pokemon"],
    },
    "semura_(character)": {
        "character": ["semura_(character)"],
        "trigger": ["semura \\(character\\), mythology"],
    },
    "melissa_(hipcat)": {
        "character": ["melissa_(hipcat)"],
        "trigger": ["melissa \\(hipcat\\), mythology"],
    },
    "nata_rivermane": {
        "character": ["nata_rivermane"],
        "trigger": ["nata rivermane, warcraft"],
    },
    "keiran_tracey": {
        "character": ["keiran_tracey"],
        "trigger": ["keiran tracey, patreon"],
    },
    "dransvitry": {
        "character": ["dransvitry"],
        "trigger": ["dransvitry, source filmmaker"],
    },
    "android_21": {"character": ["android_21"], "trigger": ["android 21, dragon ball"]},
    "lomas": {"character": ["lomas"], "trigger": ["lomas, mythology"]},
    "jorge_san_nicolas": {
        "character": ["jorge_san_nicolas"],
        "trigger": ["jorge san nicolas, texnatsu"],
    },
    "elliot_(fuf)": {
        "character": ["elliot_(fuf)"],
        "trigger": ["elliot \\(fuf\\), pokemon"],
    },
    "aevoa": {"character": ["aevoa"], "trigger": ["aevoa, slime rancher"]},
    "agatha_vulpes": {
        "character": ["agatha_vulpes"],
        "trigger": ["agatha vulpes, mythology"],
    },
    "dust_(mewgle)": {
        "character": ["dust_(mewgle)"],
        "trigger": ["dust \\(mewgle\\), no nut november"],
    },
    "flip_(flipyart)": {
        "character": ["flip_(flipyart)"],
        "trigger": ["flip \\(flipyart\\), mythology"],
    },
    "dante_(101_dalmatians)": {
        "character": ["dante_(101_dalmatians)"],
        "trigger": ["dante \\(101 dalmatians\\), disney"],
    },
    "veronica_(securipun)": {
        "character": ["veronica_(securipun)"],
        "trigger": ["veronica \\(securipun\\), mythology"],
    },
    "grammeowster_chef": {
        "character": ["grammeowster_chef"],
        "trigger": ["grammeowster chef, monster hunter"],
    },
    "terry_bat": {"character": ["terry_bat"], "trigger": ["terry bat, femboy hooters"]},
    "noz_orlok": {"character": ["noz_orlok"], "trigger": ["noz orlok, mythology"]},
    "vanilla_(buta99)": {
        "character": ["vanilla_(buta99)"],
        "trigger": ["vanilla \\(buta99\\), christmas"],
    },
    "michelle_(xxsparcoxx)": {
        "character": ["michelle_(xxsparcoxx)"],
        "trigger": ["michelle \\(xxsparcoxx\\), christmas"],
    },
    "raina_(goopyarts)": {
        "character": ["raina_(goopyarts)"],
        "trigger": ["raina \\(goopyarts\\), pokemon"],
    },
    "keisatsu_dog_(sususuigi)": {
        "character": ["keisatsu_dog_(sususuigi)"],
        "trigger": ["keisatsu dog \\(sususuigi\\), meme clothing"],
    },
    "yuki_(side_b)": {
        "character": ["yuki_(side_b)"],
        "trigger": ["yuki \\(side b\\), pokemon"],
    },
    "wendy_(bugzilla)": {
        "character": ["wendy_(bugzilla)"],
        "trigger": ["wendy \\(bugzilla\\), team cherry"],
    },
    "kralex": {"character": ["kralex"], "trigger": ["kralex, persona \\(series\\)"]},
    "spearmaster_(rain_world)": {
        "character": ["spearmaster_(rain_world)"],
        "trigger": ["spearmaster \\(rain world\\), videocult"],
    },
    "penny_(pokemon)": {
        "character": ["penny_(pokemon)"],
        "trigger": ["penny \\(pokemon\\), pokemon"],
    },
    "himalaya_(hima_nsfw)": {
        "character": ["himalaya_(hima_nsfw)"],
        "trigger": ["himalaya \\(hima nsfw\\), christmas"],
    },
    "naafiri_(lol)": {
        "character": ["naafiri_(lol)"],
        "trigger": ["naafiri \\(lol\\), riot games"],
    },
    "qinglong_(tas)": {
        "character": ["qinglong_(tas)"],
        "trigger": ["qinglong \\(tas\\), lifewonders"],
    },
    "leshana": {"character": ["leshana"], "trigger": ["leshana, mythology"]},
    "equustra_(ecmajor)": {
        "character": ["equustra_(ecmajor)"],
        "trigger": ["equustra \\(ecmajor\\), mythology"],
    },
    "waffle_(ashwaffles)": {
        "character": ["waffle_(ashwaffles)"],
        "trigger": ["waffle \\(ashwaffles\\), mythology"],
    },
    "ann_gustave": {
        "character": ["ann_gustave"],
        "trigger": ["ann gustave, las lindas"],
    },
    "hermione_granger": {
        "character": ["hermione_granger"],
        "trigger": ["hermione granger, harry potter \\(series\\)"],
    },
    "sparkster": {
        "character": ["sparkster"],
        "trigger": ["sparkster, rocket knight adventures"],
    },
    "diana_linda": {
        "character": ["diana_linda"],
        "trigger": ["diana linda, las lindas"],
    },
    "loopy_(loopy_de_loop)": {
        "character": ["loopy_(loopy_de_loop)"],
        "trigger": ["loopy \\(loopy de loop\\), loopy de loop"],
    },
    "kouryuu": {"character": ["kouryuu"], "trigger": ["kouryuu, mythology"]},
    "fatima": {"character": ["fatima"], "trigger": ["fatima, mythology"]},
    "tony_tony_chopper_(horn_point_form)": {
        "character": ["tony_tony_chopper_(horn_point_form)"],
        "trigger": ["tony tony chopper \\(horn point form\\), one piece"],
    },
    "igneous_rock_(mlp)": {
        "character": ["igneous_rock_(mlp)"],
        "trigger": ["igneous rock \\(mlp\\), my little pony"],
    },
    "malk": {"character": ["malk"], "trigger": ["malk, mythology"]},
    "flip_bunny": {"character": ["flip_bunny"], "trigger": ["flip bunny, nintendo"]},
    "akkla": {"character": ["akkla"], "trigger": ["akkla, mythology"]},
    "alice_hill": {"character": ["alice_hill"], "trigger": ["alice hill, mass effect"]},
    "bobby_frederick": {
        "character": ["bobby_frederick"],
        "trigger": ["bobby frederick, dreamkeepers"],
    },
    "vivian_rose": {
        "character": ["vivian_rose"],
        "trigger": ["vivian rose, christmas"],
    },
    "kiggles": {"character": ["kiggles"], "trigger": ["kiggles, nintendo ds family"]},
    "amy_(fvt)": {
        "character": ["amy_(fvt)"],
        "trigger": ["amy \\(fvt\\), fairies vs tentacles"],
    },
    "tantabus": {"character": ["tantabus"], "trigger": ["tantabus, my little pony"]},
    "shyren": {"character": ["shyren"], "trigger": ["shyren, undertale \\(series\\)"]},
    "frankie_(lyme-slyme)": {
        "character": ["frankie_(lyme-slyme)"],
        "trigger": ["frankie \\(lyme-slyme\\), pokemon"],
    },
    "bryce_(lonewolfhowling)": {
        "character": ["bryce_(lonewolfhowling)"],
        "trigger": ["bryce \\(lonewolfhowling\\), sony interactive entertainment"],
    },
    "fru_fru": {"character": ["fru_fru"], "trigger": ["fru fru, disney"]},
    "roly_(roly)": {
        "character": ["roly_(roly)"],
        "trigger": ["roly \\(roly\\), mythology"],
    },
    "hollyleaf_(warriors)": {
        "character": ["hollyleaf_(warriors)"],
        "trigger": ["hollyleaf \\(warriors\\), warriors \\(book series\\)"],
    },
    "pinstripe_potoroo": {
        "character": ["pinstripe_potoroo"],
        "trigger": ["pinstripe potoroo, crash bandicoot \\(series\\)"],
    },
    "aria_(killalotus119)": {
        "character": ["aria_(killalotus119)"],
        "trigger": ["aria \\(killalotus119\\), pokemon"],
    },
    "mrs._henderson": {
        "character": ["mrs._henderson"],
        "trigger": ["mrs. henderson, pokemon"],
    },
    "exorcist_(hazbin_hotel)": {
        "character": ["exorcist_(hazbin_hotel)"],
        "trigger": ["exorcist \\(hazbin hotel\\), hazbin hotel"],
    },
    "muffin_(themuffinly)": {
        "character": ["muffin_(themuffinly)"],
        "trigger": ["muffin \\(themuffinly\\), nintendo"],
    },
    "dawkins_(101_dalmatians)": {
        "character": ["dawkins_(101_dalmatians)"],
        "trigger": ["dawkins \\(101 dalmatians\\), disney"],
    },
    "reverend_(ratte)": {
        "character": ["reverend_(ratte)"],
        "trigger": ["reverend \\(ratte\\), legacy \\(ratte\\)"],
    },
    "horus_wild": {"character": ["horus_wild"], "trigger": ["horus wild, mythology"]},
    "oz_(buxombalrog)": {
        "character": ["oz_(buxombalrog)"],
        "trigger": ["oz \\(buxombalrog\\), mythology"],
    },
    "iggy_(vanzard)": {
        "character": ["iggy_(vanzard)"],
        "trigger": ["iggy \\(vanzard\\), pokemon"],
    },
    "trixie_heeler": {
        "character": ["trixie_heeler"],
        "trigger": ["trixie heeler, bluey \\(series\\)"],
    },
    "drip_(dripponi)": {
        "character": ["drip_(dripponi)"],
        "trigger": ["drip \\(dripponi\\), mythology"],
    },
    "mackenzie_border_collie": {
        "character": ["mackenzie_border_collie"],
        "trigger": ["mackenzie border collie, bluey \\(series\\)"],
    },
    "captain_grime": {
        "character": ["captain_grime"],
        "trigger": ["captain grime, disney"],
    },
    "gothabelle": {
        "character": ["gothabelle"],
        "trigger": ["gothabelle, animal crossing"],
    },
    "axel_(lazydergenboi)": {
        "character": ["axel_(lazydergenboi)"],
        "trigger": ["axel \\(lazydergenboi\\), mythology"],
    },
    "fyixa_(fyixen)": {
        "character": ["fyixa_(fyixen)"],
        "trigger": ["fyixa \\(fyixen\\), jeep"],
    },
    "heiko_(domasarts)": {
        "character": ["heiko_(domasarts)"],
        "trigger": ["heiko \\(domasarts\\), twitter"],
    },
    "shybun": {"character": ["shybun"], "trigger": ["shybun, studio trigger"]},
    "dallas_burnside_(forestdale)": {
        "character": ["dallas_burnside_(forestdale)"],
        "trigger": ["dallas burnside \\(forestdale\\), forestdale"],
    },
    "vranda_von_bat_(blarf022)": {
        "character": ["vranda_von_bat_(blarf022)"],
        "trigger": ["vranda von bat \\(blarf022\\), twitter"],
    },
    "twile": {"character": ["twile"], "trigger": ["twile, mythology"]},
    "ethan_white": {
        "character": ["ethan_white"],
        "trigger": ["ethan white, clubstripes"],
    },
    "kitty_vanilji": {
        "character": ["kitty_vanilji"],
        "trigger": ["kitty vanilji, mythology"],
    },
    "constance_jotkowska_(coyotek)": {
        "character": ["constance_jotkowska_(coyotek)"],
        "trigger": ["constance jotkowska \\(coyotek\\), mythology"],
    },
    "xlr8": {"character": ["xlr8"], "trigger": ["xlr8, cartoon network"]},
    "gaon": {"character": ["gaon"], "trigger": ["gaon, kaiketsu zorori"]},
    "conquering_storm": {
        "character": ["conquering_storm"],
        "trigger": ["conquering storm, sonic the hedgehog \\(series\\)"],
    },
    "boots_(dora_the_explorer)": {
        "character": ["boots_(dora_the_explorer)"],
        "trigger": ["boots \\(dora the explorer\\), dora the explorer"],
    },
    "amy_wong": {"character": ["amy_wong"], "trigger": ["amy wong, comedy central"]},
    "marie_(aristocats)": {
        "character": ["marie_(aristocats)"],
        "trigger": ["marie \\(aristocats\\), disney"],
    },
    "ottah": {"character": ["ottah"], "trigger": ["ottah, star wars"]},
    "madarao_(kekkaishi)": {
        "character": ["madarao_(kekkaishi)"],
        "trigger": ["madarao \\(kekkaishi\\), kekkaishi"],
    },
    "derek_(kitaness)": {
        "character": ["derek_(kitaness)"],
        "trigger": ["derek \\(kitaness\\), the mysteries of alfred hedgehog"],
    },
    "mike_wazowski": {
        "character": ["mike_wazowski"],
        "trigger": ["mike wazowski, pixar"],
    },
    "nilla": {"character": ["nilla"], "trigger": ["nilla, nintendo"]},
    "mauro_skyles": {
        "character": ["mauro_skyles"],
        "trigger": ["mauro skyles, mythology"],
    },
    "abriika": {"character": ["abriika"], "trigger": ["abriika, haa"]},
    "lady_wolf_(arbuzbudesh)": {
        "character": ["lady_wolf_(arbuzbudesh)"],
        "trigger": ["lady wolf \\(arbuzbudesh\\), mythology"],
    },
    "jasmin_(donutella)": {
        "character": ["jasmin_(donutella)"],
        "trigger": ["jasmin \\(donutella\\), pokemon"],
    },
    "arunira": {"character": ["arunira"], "trigger": ["arunira, christmas"]},
    "forsburn": {"character": ["forsburn"], "trigger": ["forsburn, rivals of aether"]},
    "fhtng_the_unspeakable": {
        "character": ["fhtng_the_unspeakable"],
        "trigger": ["fhtng the unspeakable, them's fightin' herds"],
    },
    "female_frisk_(undertale)": {
        "character": ["female_frisk_(undertale)"],
        "trigger": ["female frisk \\(undertale\\), undertale \\(series\\)"],
    },
    "sir_squiggles_(character)": {
        "character": ["sir_squiggles_(character)"],
        "trigger": ["sir squiggles \\(character\\), mythology"],
    },
    "donald_trump": {
        "character": ["donald_trump"],
        "trigger": ["donald trump, nintendo"],
    },
    "sybil_mccready": {
        "character": ["sybil_mccready"],
        "trigger": ["sybil mccready, halloween"],
    },
    "night_owl_(creatures_of_the_night)": {
        "character": ["night_owl_(creatures_of_the_night)"],
        "trigger": ["night owl \\(creatures of the night\\), creatures of the night"],
    },
    "ricky_(fuze)": {
        "character": ["ricky_(fuze)"],
        "trigger": ["ricky \\(fuze\\), pokemon"],
    },
    "neogoldwing": {
        "character": ["neogoldwing"],
        "trigger": ["neogoldwing, mythology"],
    },
    "iselda_(hollow_knight)": {
        "character": ["iselda_(hollow_knight)"],
        "trigger": ["iselda \\(hollow knight\\), team cherry"],
    },
    "lucia_(paledrake)": {
        "character": ["lucia_(paledrake)"],
        "trigger": ["lucia \\(paledrake\\), mythology"],
    },
    "erika_(ambris)": {
        "character": ["erika_(ambris)"],
        "trigger": ["erika \\(ambris\\), hands-free bubble tea"],
    },
    "alex_(carpetwurm)": {
        "character": ["alex_(carpetwurm)"],
        "trigger": ["alex \\(carpetwurm\\), nintendo"],
    },
    "nyla_(whitekitteh)": {
        "character": ["nyla_(whitekitteh)"],
        "trigger": ["nyla \\(whitekitteh\\), mythology"],
    },
    "draxius": {"character": ["draxius"], "trigger": ["draxius, mythology"]},
    "rudolph_holiday": {
        "character": ["rudolph_holiday"],
        "trigger": ["rudolph holiday, undertale \\(series\\)"],
    },
    "snofu_(character)": {
        "character": ["snofu_(character)"],
        "trigger": ["snofu \\(character\\), mythology"],
    },
    "avey_(avey_aveon)": {
        "character": ["avey_(avey_aveon)"],
        "trigger": ["avey \\(avey aveon\\), christmas"],
    },
    "katlin_perkins": {
        "character": ["katlin_perkins"],
        "trigger": ["katlin perkins, pixile studios"],
    },
    "geld_(that_time_i_got_reincarnated_as_a_slime)": {
        "character": ["geld_(that_time_i_got_reincarnated_as_a_slime)"],
        "trigger": [
            "geld \\(that time i got reincarnated as a slime\\), that time i got reincarnated as a slime"
        ],
    },
    "johnny_bunny": {
        "character": ["johnny_bunny"],
        "trigger": ["johnny bunny, halloween"],
    },
    "glaze_(thepianofurry)": {
        "character": ["glaze_(thepianofurry)"],
        "trigger": ["glaze \\(thepianofurry\\), my little pony"],
    },
    "hilda_the_huntress": {
        "character": ["hilda_the_huntress"],
        "trigger": ["hilda the huntress, realm royale"],
    },
    "lou_(thekite)": {
        "character": ["lou_(thekite)"],
        "trigger": ["lou \\(thekite\\), pokemon"],
    },
    "einarr_(personalami)": {
        "character": ["einarr_(personalami)"],
        "trigger": ["einarr \\(personalami\\), mythology"],
    },
    "jasper_(pizzacow)": {
        "character": ["jasper_(pizzacow)"],
        "trigger": ["jasper \\(pizzacow\\), mythology"],
    },
    "basian": {"character": ["basian"], "trigger": ["basian, mythology"]},
    "crumb_(buizel)": {
        "character": ["crumb_(buizel)"],
        "trigger": ["crumb \\(buizel\\), pokemon"],
    },
    "fatal_(fatal_dx)": {
        "character": ["fatal_(fatal_dx)"],
        "trigger": ["fatal \\(fatal dx\\), pokemon"],
    },
    "angel_(code01)": {
        "character": ["angel_(code01)"],
        "trigger": ["angel \\(code01\\)"],
    },
    "l33t_(labbit)": {
        "character": ["l33t_(labbit)"],
        "trigger": ["l33t \\(labbit\\), hazbin hotel"],
    },
    "mhicky_(mhicky93)": {
        "character": ["mhicky_(mhicky93)"],
        "trigger": ["mhicky \\(mhicky93\\), pokemon"],
    },
    "nemona_(pokemon)": {
        "character": ["nemona_(pokemon)"],
        "trigger": ["nemona \\(pokemon\\), pokemon"],
    },
    "ott_(brok_the_investigator)": {
        "character": ["ott_(brok_the_investigator)"],
        "trigger": ["ott \\(brok the investigator\\), brok the investigator"],
    },
    "lune_(lunesnowtail)": {
        "character": ["lune_(lunesnowtail)"],
        "trigger": ["lune \\(lunesnowtail\\), dust: an elysian tail"],
    },
    "morvay_(nu:_carnival)": {
        "character": ["morvay_(nu:_carnival)"],
        "trigger": ["morvay \\(nu: carnival\\), nu: carnival"],
    },
    "rebecca_(cyberpunk_edgerunners)": {
        "character": ["rebecca_(cyberpunk_edgerunners)"],
        "trigger": ["rebecca \\(cyberpunk edgerunners\\), cyberpunk edgerunners"],
    },
    "toriel_(dogzeela)": {
        "character": ["toriel_(dogzeela)"],
        "trigger": ["toriel \\(dogzeela\\), undertale \\(series\\)"],
    },
    "jennifer_(divine_acid)": {
        "character": ["jennifer_(divine_acid)"],
        "trigger": ["jennifer \\(divine acid\\), divine acid"],
    },
    "fender": {"character": ["fender"], "trigger": ["fender, furaffinity"]},
    "iridium": {"character": ["iridium"], "trigger": ["iridium, microsoft paint"]},
    "the_doctor_(doctor_who)": {
        "character": ["the_doctor_(doctor_who)"],
        "trigger": ["the doctor \\(doctor who\\), doctor who"],
    },
    "lah_(sonic)": {"character": ["lah_(sonic)"], "trigger": ["lah \\(sonic\\), sega"]},
    "piccolo": {"character": ["piccolo"], "trigger": ["piccolo, dragon ball"]},
    "helsy_(helsy)": {
        "character": ["helsy_(helsy)"],
        "trigger": ["helsy \\(helsy\\), mythology"],
    },
    "molly_fullin": {
        "character": ["molly_fullin"],
        "trigger": ["molly fullin, christmas"],
    },
    "scratte_(ice_age)": {
        "character": ["scratte_(ice_age)"],
        "trigger": ["scratte \\(ice age\\), ice age \\(series\\)"],
    },
    "thel_'vadam": {
        "character": ["thel_'vadam"],
        "trigger": ["thel 'vadam, halo \\(series\\)"],
    },
    "avalon": {"character": ["avalon"], "trigger": ["avalon, mythology"]},
    "rokuke_shiba_(character)": {
        "character": ["rokuke_shiba_(character)"],
        "trigger": ["rokuke shiba \\(character\\), pokemon"],
    },
    "sakido_elexion": {
        "character": ["sakido_elexion"],
        "trigger": ["sakido elexion, slightly damned"],
    },
    "john_(meesh)": {
        "character": ["john_(meesh)"],
        "trigger": ["john \\(meesh\\), little buddy"],
    },
    "ifrit_(final_fantasy)": {
        "character": ["ifrit_(final_fantasy)"],
        "trigger": ["ifrit \\(final fantasy\\), square enix"],
    },
    "eddie_(evane)": {
        "character": ["eddie_(evane)"],
        "trigger": ["eddie \\(evane\\), evane"],
    },
    "grace_kaiser": {
        "character": ["grace_kaiser"],
        "trigger": ["grace kaiser, mayfield"],
    },
    "markus_(kadath)": {
        "character": ["markus_(kadath)"],
        "trigger": ["markus \\(kadath\\), patreon"],
    },
    "hydryl": {"character": ["hydryl"], "trigger": ["hydryl, mythology"]},
    "fill": {"character": ["fill"], "trigger": ["fill, nintendo"]},
    "bramblestar_(warriors)": {
        "character": ["bramblestar_(warriors)"],
        "trigger": ["bramblestar \\(warriors\\), warriors \\(book series\\)"],
    },
    "assumi": {"character": ["assumi"], "trigger": ["assumi, warcraft"]},
    "snowy_(duckdraw)": {
        "character": ["snowy_(duckdraw)"],
        "trigger": ["snowy \\(duckdraw\\), christmas"],
    },
    "goblin_princess": {
        "character": ["goblin_princess"],
        "trigger": ["goblin princess, towergirls"],
    },
    "zoe_heartfields": {
        "character": ["zoe_heartfields"],
        "trigger": ["zoe heartfields, patreon"],
    },
    "tracy_vale": {"character": ["tracy_vale"], "trigger": ["tracy vale, christmas"]},
    "bill_(skybluefox)": {
        "character": ["bill_(skybluefox)"],
        "trigger": ["bill \\(skybluefox\\), pokemon"],
    },
    "arawn_(howlfeiwolf)": {
        "character": ["arawn_(howlfeiwolf)"],
        "trigger": ["arawn \\(howlfeiwolf\\), mythology"],
    },
    "reaper_(overwatch)": {
        "character": ["reaper_(overwatch)"],
        "trigger": ["reaper \\(overwatch\\), overwatch"],
    },
    "island_(character)": {
        "character": ["island_(character)"],
        "trigger": ["island \\(character\\), nintendo"],
    },
    "auguscus_acilcolus": {
        "character": ["auguscus_acilcolus"],
        "trigger": ["auguscus acilcolus, mass effect"],
    },
    "moki_(character)": {
        "character": ["moki_(character)"],
        "trigger": ["moki \\(character\\), halloween"],
    },
    "dodger_(creatures_of_the_night)": {
        "character": ["dodger_(creatures_of_the_night)"],
        "trigger": ["dodger \\(creatures of the night\\), creatures of the night"],
    },
    "renard_(homura_kasuka)": {
        "character": ["renard_(homura_kasuka)"],
        "trigger": ["renard \\(homura kasuka\\), drecom"],
    },
    "shakes_heartwood": {
        "character": ["shakes_heartwood"],
        "trigger": ["shakes heartwood, my little pony"],
    },
    "azura_(azura_inalis)": {
        "character": ["azura_(azura_inalis)"],
        "trigger": ["azura \\(azura inalis\\), mythology"],
    },
    "pyra_(xenoblade)": {
        "character": ["pyra_(xenoblade)"],
        "trigger": ["pyra \\(xenoblade\\), xenoblade \\(series\\)"],
    },
    "star_tracker_(mlp)": {
        "character": ["star_tracker_(mlp)"],
        "trigger": ["star tracker \\(mlp\\), my little pony"],
    },
    "yulia_(bakedbunny)": {
        "character": ["yulia_(bakedbunny)"],
        "trigger": ["yulia \\(bakedbunny\\), mythology"],
    },
    "elaine_(pokemon)": {
        "character": ["elaine_(pokemon)"],
        "trigger": ["elaine \\(pokemon\\), pokemon: let's go"],
    },
    "pizza_rabbit_(rabblet)": {
        "character": ["pizza_rabbit_(rabblet)"],
        "trigger": ["pizza rabbit \\(rabblet\\), easter"],
    },
    "zaphira_(zummeng)": {
        "character": ["zaphira_(zummeng)"],
        "trigger": ["zaphira \\(zummeng\\), patreon"],
    },
    "lux_(namoke)": {
        "character": ["lux_(namoke)"],
        "trigger": ["lux \\(namoke\\), valentine's day"],
    },
    "ruzeth": {"character": ["ruzeth"], "trigger": ["ruzeth, my little pony"]},
    "mohuko_(komenuka_inaho)": {
        "character": ["mohuko_(komenuka_inaho)"],
        "trigger": ["mohuko \\(komenuka inaho\\), mythology"],
    },
    "august_moon": {
        "character": ["august_moon"],
        "trigger": ["august moon, christmas"],
    },
    "sachii_(resachii)": {
        "character": ["sachii_(resachii)"],
        "trigger": ["sachii \\(resachii\\), pokemon"],
    },
    "samantha_(infinity_train)": {
        "character": ["samantha_(infinity_train)"],
        "trigger": ["samantha \\(infinity train\\), infinity train"],
    },
    "repzzmonster_(character)": {
        "character": ["repzzmonster_(character)"],
        "trigger": ["repzzmonster \\(character\\), mythology"],
    },
    "hald_(manadezimon)": {
        "character": ["hald_(manadezimon)"],
        "trigger": ["hald \\(manadezimon\\), halloween"],
    },
    "ryn_(stargazer)": {
        "character": ["ryn_(stargazer)"],
        "trigger": ["ryn \\(stargazer\\), to be continued"],
    },
    "axey_(wazzaldorp)": {
        "character": ["axey_(wazzaldorp)"],
        "trigger": ["axey \\(wazzaldorp\\), real axolotl hours"],
    },
    "multum": {"character": ["multum"], "trigger": ["multum, mythology"]},
    "edgar_vladilisitsa": {
        "character": ["edgar_vladilisitsa"],
        "trigger": ["edgar vladilisitsa, shut \\(meme\\)"],
    },
    "five_pebbles_(rain_world)": {
        "character": ["five_pebbles_(rain_world)"],
        "trigger": ["five pebbles \\(rain world\\), videocult"],
    },
    "kiwi_(viroveteruscy)": {
        "character": ["kiwi_(viroveteruscy)"],
        "trigger": ["kiwi \\(viroveteruscy\\), warning cream filled"],
    },
    "cait_(world_of_ruan)": {
        "character": ["cait_(world_of_ruan)"],
        "trigger": ["cait \\(world of ruan\\), world of ruan"],
    },
    "nyanlathotep_(sucker_for_love)": {
        "character": ["nyanlathotep_(sucker_for_love)"],
        "trigger": ["nyanlathotep \\(sucker for love\\), sucker for love"],
    },
    "aevyn": {"character": ["aevyn"], "trigger": ["aevyn, nintendo"]},
    "zeezee_murdock": {
        "character": ["zeezee_murdock"],
        "trigger": ["zeezee murdock, pokemon"],
    },
    "sonia_(lonnyk)": {
        "character": ["sonia_(lonnyk)"],
        "trigger": ["sonia \\(lonnyk\\), 101 dalmatians"],
    },
    "savannah_reed": {
        "character": ["savannah_reed"],
        "trigger": ["savannah reed, hasbro"],
    },
    "ruth_(peculiart)": {
        "character": ["ruth_(peculiart)"],
        "trigger": ["ruth \\(peculiart\\), bun bash"],
    },
    "krystal_(charleyfox)": {
        "character": ["krystal_(charleyfox)"],
        "trigger": ["krystal \\(charleyfox\\), star fox"],
    },
    "piglet": {"character": ["piglet"], "trigger": ["piglet, disney"]},
    "zalgo_(creepypasta)": {
        "character": ["zalgo_(creepypasta)"],
        "trigger": ["zalgo \\(creepypasta\\), lovecraftian \\(genre\\)"],
    },
    "toothy_(htf)": {
        "character": ["toothy_(htf)"],
        "trigger": ["toothy \\(htf\\), happy tree friends"],
    },
    "kamui_(hitsunekun)": {
        "character": ["kamui_(hitsunekun)"],
        "trigger": ["kamui \\(hitsunekun\\), square enix"],
    },
    "hexerade": {"character": ["hexerade"], "trigger": ["hexerade, e621"]},
    "rose_lalonde": {
        "character": ["rose_lalonde"],
        "trigger": ["rose lalonde, homestuck"],
    },
    "frionella": {"character": ["frionella"], "trigger": ["frionella, mythology"]},
    "trigger_(trigger12)": {
        "character": ["trigger_(trigger12)"],
        "trigger": ["trigger \\(trigger12\\), trigger12"],
    },
    "scylla_(coc)": {
        "character": ["scylla_(coc)"],
        "trigger": ["scylla \\(coc\\), corruption of champions"],
    },
    "waddles_(gravity_falls)": {
        "character": ["waddles_(gravity_falls)"],
        "trigger": ["waddles \\(gravity falls\\), disney"],
    },
    "nargle_(nargleflex)": {
        "character": ["nargle_(nargleflex)"],
        "trigger": ["nargle \\(nargleflex\\), pokemon"],
    },
    "kamuri": {"character": ["kamuri"], "trigger": ["kamuri, mythology"]},
    "tegon_(dsc85)": {
        "character": ["tegon_(dsc85)"],
        "trigger": ["tegon \\(dsc85\\), mythology"],
    },
    "solaxe": {"character": ["solaxe"], "trigger": ["solaxe, mythology"]},
    "deirdrefang": {
        "character": ["deirdrefang"],
        "trigger": ["deirdrefang, second life"],
    },
    "leah_(lipton)": {
        "character": ["leah_(lipton)"],
        "trigger": ["leah \\(lipton\\), christmas"],
    },
    "vanessa_(sandwich-anomaly)": {
        "character": ["vanessa_(sandwich-anomaly)"],
        "trigger": ["vanessa \\(sandwich-anomaly\\), invader zim"],
    },
    "flash_(donutella)": {
        "character": ["flash_(donutella)"],
        "trigger": ["flash \\(donutella\\), pokemon"],
    },
    "kayla_(zoophobia)": {
        "character": ["kayla_(zoophobia)"],
        "trigger": ["kayla \\(zoophobia\\), zoophobia"],
    },
    "chun-ni": {"character": ["chun-ni"], "trigger": ["chun-ni, miracle star"]},
    "velvela": {"character": ["velvela"], "trigger": ["velvela, pokemon"]},
    "party_favor_(mlp)": {
        "character": ["party_favor_(mlp)"],
        "trigger": ["party favor \\(mlp\\), my little pony"],
    },
    "crywolf": {"character": ["crywolf"], "trigger": ["crywolf, mythology"]},
    "peggy_patterson": {
        "character": ["peggy_patterson"],
        "trigger": ["peggy patterson, new year"],
    },
    "tunny": {"character": ["tunny"], "trigger": ["tunny, netflix"]},
    "centipeetle": {
        "character": ["centipeetle"],
        "trigger": ["centipeetle, cartoon network"],
    },
    "lyx_(lynxer)": {
        "character": ["lyx_(lynxer)"],
        "trigger": ["lyx \\(lynxer\\), snapchat"],
    },
    "amprat": {"character": ["amprat"], "trigger": ["amprat, riot games"]},
    "saewin": {"character": ["saewin"], "trigger": ["saewin, mythology"]},
    "lucky_(luckyabsol)": {
        "character": ["lucky_(luckyabsol)"],
        "trigger": ["lucky \\(luckyabsol\\), pokemon"],
    },
    "marble_(gittonsxv)": {
        "character": ["marble_(gittonsxv)"],
        "trigger": ["marble \\(gittonsxv\\), spyro the dragon"],
    },
    "jill_(chris13131415)": {
        "character": ["jill_(chris13131415)"],
        "trigger": ["jill \\(chris13131415\\), mythology"],
    },
    "dj_shark_(dj_sharkowski)": {
        "character": ["dj_shark_(dj_sharkowski)"],
        "trigger": ["dj shark \\(dj sharkowski\\), cartoon network"],
    },
    "kiku_(white_knight19)": {
        "character": ["kiku_(white_knight19)"],
        "trigger": ["kiku \\(white knight19\\), warcraft"],
    },
    "seth_(sethpup)": {
        "character": ["seth_(sethpup)"],
        "trigger": ["seth \\(sethpup\\), no nut november"],
    },
    "redfiery": {"character": ["redfiery"], "trigger": ["redfiery, mythology"]},
    "riko_(made_in_abyss)": {
        "character": ["riko_(made_in_abyss)"],
        "trigger": ["riko \\(made in abyss\\), made in abyss"],
    },
    "alissa_(alasou)": {
        "character": ["alissa_(alasou)"],
        "trigger": ["alissa \\(alasou\\), nintendo"],
    },
    "nomsi_(character)": {
        "character": ["nomsi_(character)"],
        "trigger": ["nomsi \\(character\\), christmas"],
    },
    "mizett_(doneru)": {
        "character": ["mizett_(doneru)"],
        "trigger": ["mizett \\(doneru\\), mythology"],
    },
    "ashe_(starshippizza)": {
        "character": ["ashe_(starshippizza)"],
        "trigger": ["ashe \\(starshippizza\\), nintendo"],
    },
    "rikki_landon": {
        "character": ["rikki_landon"],
        "trigger": ["rikki landon, mythology"],
    },
    "luca_(patto)": {
        "character": ["luca_(patto)"],
        "trigger": ["luca \\(patto\\), mythology"],
    },
    "amelie_(jinx_doodle)": {
        "character": ["amelie_(jinx_doodle)"],
        "trigger": ["amelie \\(jinx doodle\\), nintendo"],
    },
    "amelia_(petruz)": {
        "character": ["amelia_(petruz)"],
        "trigger": ["amelia \\(petruz\\), petruz \\(copyright\\)"],
    },
    "ty_(appleseed)": {
        "character": ["ty_(appleseed)"],
        "trigger": ["ty \\(appleseed\\), disney"],
    },
    "drake_inrelal": {
        "character": ["drake_inrelal"],
        "trigger": ["drake inrelal, mythology"],
    },
    "milo_(cherryfox73)": {
        "character": ["milo_(cherryfox73)"],
        "trigger": ["milo \\(cherryfox73\\), nintendo"],
    },
    "anna_(ayaka)": {"character": ["anna_(ayaka)"], "trigger": ["anna \\(ayaka\\)"]},
    "basedvulpine_(character)": {
        "character": ["basedvulpine_(character)"],
        "trigger": ["basedvulpine \\(character\\), blender \\(software\\)"],
    },
    "lily_mari": {
        "character": ["lily_mari"],
        "trigger": ["lily mari, valentine's day"],
    },
    "jenny_(insomniacovrlrd)": {
        "character": ["jenny_(insomniacovrlrd)"],
        "trigger": ["jenny \\(insomniacovrlrd\\), passiontail isle"],
    },
    "sunny_macchiato": {
        "character": ["sunny_macchiato"],
        "trigger": ["sunny macchiato, christmas"],
    },
    "fred_(fredrick_brennan)": {
        "character": ["fred_(fredrick_brennan)"],
        "trigger": ["fred \\(fredrick brennan\\), linux"],
    },
    "warfare_lop": {
        "character": ["warfare_lop"],
        "trigger": ["warfare lop, star wars visions"],
    },
    "danny_(m0ffedup)": {
        "character": ["danny_(m0ffedup)"],
        "trigger": ["danny \\(m0ffedup\\), nintendo"],
    },
    "jei_laule": {
        "character": ["jei_laule"],
        "trigger": ["jei laule, east asian mythology"],
    },
    "gourmand_(rain_world)": {
        "character": ["gourmand_(rain_world)"],
        "trigger": ["gourmand \\(rain world\\), videocult"],
    },
    "riko_sakari": {
        "character": ["riko_sakari"],
        "trigger": ["riko sakari, mythology"],
    },
    "shayla_the_pink_mouse": {
        "character": ["shayla_the_pink_mouse"],
        "trigger": ["shayla the pink mouse, mythology"],
    },
    "tymbre": {"character": ["tymbre"], "trigger": ["tymbre, mythology"]},
    "captain_neyla": {
        "character": ["captain_neyla"],
        "trigger": ["captain neyla, sucker punch productions"],
    },
    "rikamon": {"character": ["rikamon"], "trigger": ["rikamon, digimon"]},
    "miley_mouse": {
        "character": ["miley_mouse"],
        "trigger": ["miley mouse, mythology"],
    },
    "vault_dwellers_(fallout)": {
        "character": ["vault_dwellers_(fallout)"],
        "trigger": ["vault dwellers \\(fallout\\), fallout"],
    },
    "gaslight": {"character": ["gaslight"], "trigger": ["gaslight, gaslightdog"]},
    "jenn_(zp92)": {
        "character": ["jenn_(zp92)"],
        "trigger": ["jenn \\(zp92\\), source filmmaker"],
    },
    "shifty_(htf)": {
        "character": ["shifty_(htf)"],
        "trigger": ["shifty \\(htf\\), happy tree friends"],
    },
    "scynt": {"character": ["scynt"], "trigger": ["scynt, my little pony"]},
    "rorikemo_(j7w)": {
        "character": ["rorikemo_(j7w)"],
        "trigger": ["rorikemo \\(j7w\\), mythology"],
    },
    "alex_(jrbart)": {
        "character": ["alex_(jrbart)"],
        "trigger": ["alex \\(jrbart\\), mythology"],
    },
    "cypher": {"character": ["cypher"], "trigger": ["cypher, mythology"]},
    "stoaty": {"character": ["stoaty"], "trigger": ["stoaty, the shadow of light"]},
    "nol": {"character": ["nol"], "trigger": ["nol, mythology"]},
    "claire_wheeler": {
        "character": ["claire_wheeler"],
        "trigger": ["claire wheeler, disney"],
    },
    "kage6415": {"character": ["kage6415"], "trigger": ["kage6415, mythology"]},
    "tom_(ehs)": {"character": ["tom_(ehs)"], "trigger": ["tom \\(ehs\\), pringles"]},
    "coco_(doctor_lollipop)": {
        "character": ["coco_(doctor_lollipop)"],
        "trigger": ["coco \\(doctor lollipop\\), cartoon hangover"],
    },
    "cackletta": {"character": ["cackletta"], "trigger": ["cackletta, mario bros"]},
    "keinos": {"character": ["keinos"], "trigger": ["keinos, mythology"]},
    "jack_darby": {"character": ["jack_darby"], "trigger": ["jack darby, takara tomy"]},
    "gail_(badsheep)": {
        "character": ["gail_(badsheep)"],
        "trigger": ["gail \\(badsheep\\), pokemon"],
    },
    "nora_(elfdrago)": {
        "character": ["nora_(elfdrago)"],
        "trigger": ["nora \\(elfdrago\\), pokemon"],
    },
    "glory_(wof)": {
        "character": ["glory_(wof)"],
        "trigger": ["glory \\(wof\\), wings of fire"],
    },
    "ark_(shade)": {"character": ["ark_(shade)"], "trigger": ["ark \\(shade\\)"]},
    "stormy_flare_(mlp)": {
        "character": ["stormy_flare_(mlp)"],
        "trigger": ["stormy flare \\(mlp\\), my little pony"],
    },
    "prilly_(lysergide)": {
        "character": ["prilly_(lysergide)"],
        "trigger": ["prilly \\(lysergide\\), pokemon"],
    },
    "tanashi": {"character": ["tanashi"], "trigger": ["tanashi, pokemon"]},
    "gat_(gatboy)": {
        "character": ["gat_(gatboy)"],
        "trigger": ["gat \\(gatboy\\), square enix"],
    },
    "nonine": {"character": ["nonine"], "trigger": ["nonine, pokemon"]},
    "eddie_(orf)": {
        "character": ["eddie_(orf)"],
        "trigger": ["eddie \\(orf\\), halloween"],
    },
    "anglo_(anglo)": {
        "character": ["anglo_(anglo)"],
        "trigger": ["anglo \\(anglo\\), pokemon"],
    },
    "alym": {"character": ["alym"], "trigger": ["alym, mythology"]},
    "julia_woods": {
        "character": ["julia_woods"],
        "trigger": ["julia woods, book of lust"],
    },
    "lana_banana_(felino)": {
        "character": ["lana_banana_(felino)"],
        "trigger": ["lana banana \\(felino\\), nintendo"],
    },
    "sarah_fairhart": {
        "character": ["sarah_fairhart"],
        "trigger": ["sarah fairhart, greek mythology"],
    },
    "katherine_(appledees)": {
        "character": ["katherine_(appledees)"],
        "trigger": ["katherine \\(appledees\\), meme clothing"],
    },
    "noir_(yobie)": {
        "character": ["noir_(yobie)"],
        "trigger": ["noir \\(yobie\\), pokemon"],
    },
    "nighdruth_(character)": {
        "character": ["nighdruth_(character)"],
        "trigger": ["nighdruth \\(character\\), mythology"],
    },
    "leon_(pokemon)": {
        "character": ["leon_(pokemon)"],
        "trigger": ["leon \\(pokemon\\), pokemon"],
    },
    "jolt_(wm149)": {
        "character": ["jolt_(wm149)"],
        "trigger": ["jolt \\(wm149\\), pokemon"],
    },
    "lace_(lacethecutegoat)": {
        "character": ["lace_(lacethecutegoat)"],
        "trigger": ["lace \\(lacethecutegoat\\), snapchat"],
    },
    "queen_twilight_(mlp)": {
        "character": ["queen_twilight_(mlp)"],
        "trigger": ["queen twilight \\(mlp\\), my little pony"],
    },
    "kaguya_(umpherio)": {
        "character": ["kaguya_(umpherio)"],
        "trigger": ["kaguya \\(umpherio\\), mythology"],
    },
    "cay_(toxoglossa)": {
        "character": ["cay_(toxoglossa)"],
        "trigger": ["cay \\(toxoglossa\\), egyptian mythology"],
    },
    "shishiro_botan": {
        "character": ["shishiro_botan"],
        "trigger": ["shishiro botan, hololive"],
    },
    "indigo_(artca9)": {
        "character": ["indigo_(artca9)"],
        "trigger": ["indigo \\(artca9\\), jeep"],
    },
    "r.j._(brok_the_investigator)": {
        "character": ["r.j._(brok_the_investigator)"],
        "trigger": ["r.j. \\(brok the investigator\\), brok the investigator"],
    },
    "coco_(unicorn_wars)": {
        "character": ["coco_(unicorn_wars)"],
        "trigger": ["coco \\(unicorn wars\\), unicorn wars"],
    },
    "lps_675": {"character": ["lps_675"], "trigger": ["lps 675, hasbro"]},
    "omaneko_(jasdf)": {
        "character": ["omaneko_(jasdf)"],
        "trigger": ["omaneko \\(jasdf\\), japan air self-defense force"],
    },
    "javid_(dislyte)": {
        "character": ["javid_(dislyte)"],
        "trigger": ["javid \\(dislyte\\), dislyte"],
    },
    "the_emperor_(baldur's_gate)": {
        "character": ["the_emperor_(baldur's_gate)"],
        "trigger": ["the emperor \\(baldur's gate\\), electronic arts"],
    },
    "randomdragon_(character)": {
        "character": ["randomdragon_(character)"],
        "trigger": ["randomdragon \\(character\\), mythology"],
    },
    "ms._pac-man": {
        "character": ["ms._pac-man"],
        "trigger": ["ms. pac-man, pac-man \\(series\\)"],
    },
    "timtam": {"character": ["timtam"], "trigger": ["timtam, patreon"]},
    "arawn": {"character": ["arawn"], "trigger": ["arawn, mythology"]},
    "tsukiyo": {"character": ["tsukiyo"], "trigger": ["tsukiyo, mythology"]},
    "jensca": {"character": ["jensca"], "trigger": ["jensca, camp pines"]},
    "toro_inoue": {
        "character": ["toro_inoue"],
        "trigger": ["toro inoue, sony corporation"],
    },
    "hysterium": {"character": ["hysterium"], "trigger": ["hysterium, mythology"]},
    "sein_kraft": {"character": ["sein_kraft"], "trigger": ["sein kraft, mythology"]},
    "aryll": {"character": ["aryll"], "trigger": ["aryll, the legend of zelda"]},
    "delilah_(gargoyles)": {
        "character": ["delilah_(gargoyles)"],
        "trigger": ["delilah \\(gargoyles\\), disney"],
    },
    "aerys": {"character": ["aerys"], "trigger": ["aerys, mythology"]},
    "daxterdingo": {
        "character": ["daxterdingo"],
        "trigger": ["daxterdingo, scottgames"],
    },
    "julian_(kazecat)": {
        "character": ["julian_(kazecat)"],
        "trigger": ["julian \\(kazecat\\), mythology"],
    },
    "akasha_the_queen_of_pain": {
        "character": ["akasha_the_queen_of_pain"],
        "trigger": ["akasha the queen of pain, dota"],
    },
    "dal_(joelasko)": {
        "character": ["dal_(joelasko)"],
        "trigger": ["dal \\(joelasko\\), disney"],
    },
    "james_(sayuncle)": {
        "character": ["james_(sayuncle)"],
        "trigger": ["james \\(sayuncle\\), denver broncos"],
    },
    "azure_night": {
        "character": ["azure_night"],
        "trigger": ["azure night, my little pony"],
    },
    "master_oogway": {
        "character": ["master_oogway"],
        "trigger": ["master oogway, kung fu panda"],
    },
    "wendy_corduroy": {
        "character": ["wendy_corduroy"],
        "trigger": ["wendy corduroy, disney"],
    },
    "gabby_(kadath)": {
        "character": ["gabby_(kadath)"],
        "trigger": ["gabby \\(kadath\\), patreon"],
    },
    "kitchiki_(character)": {
        "character": ["kitchiki_(character)"],
        "trigger": ["kitchiki \\(character\\), mythology"],
    },
    "kade_(savestate)": {
        "character": ["kade_(savestate)"],
        "trigger": ["kade \\(savestate\\), savestate"],
    },
    "sonny_boop": {"character": ["sonny_boop"], "trigger": ["sonny boop, da silva"]},
    "sissel_(repeat)": {
        "character": ["sissel_(repeat)"],
        "trigger": ["sissel \\(repeat\\), repeat \\(visual novel\\)"],
    },
    "claudia_(averyshadydolphin)": {
        "character": ["claudia_(averyshadydolphin)"],
        "trigger": ["claudia \\(averyshadydolphin\\), mythology"],
    },
    "james_flynn": {"character": ["james_flynn"], "trigger": ["james flynn, nintendo"]},
    "luna_the_eevee": {
        "character": ["luna_the_eevee"],
        "trigger": ["luna the eevee, pokemon"],
    },
    "dancer_of_the_boreal_valley": {
        "character": ["dancer_of_the_boreal_valley"],
        "trigger": ["dancer of the boreal valley, fromsoftware"],
    },
    "martha_wakeman": {
        "character": ["martha_wakeman"],
        "trigger": ["martha wakeman, halloween"],
    },
    "sochi_(lynx)": {
        "character": ["sochi_(lynx)"],
        "trigger": ["sochi \\(lynx\\), wizards of the coast"],
    },
    "hakuna": {"character": ["hakuna"], "trigger": ["hakuna, patreon"]},
    "hati_(tas)": {
        "character": ["hati_(tas)"],
        "trigger": ["hati \\(tas\\), lifewonders"],
    },
    "carla_guzman": {
        "character": ["carla_guzman"],
        "trigger": ["carla guzman, um jammer lammy"],
    },
    "alicia_(northwynd)": {
        "character": ["alicia_(northwynd)"],
        "trigger": ["alicia \\(northwynd\\), northwynd"],
    },
    "celia_(s2-freak)": {
        "character": ["celia_(s2-freak)"],
        "trigger": ["celia \\(s2-freak\\), east asian mythology"],
    },
    "snoot_(trinity-fate62)": {
        "character": ["snoot_(trinity-fate62)"],
        "trigger": ["snoot \\(trinity-fate62\\), nintendo"],
    },
    "hun-yi_(wherewolf)": {
        "character": ["hun-yi_(wherewolf)"],
        "trigger": ["hun-yi \\(wherewolf\\), pixiv fanbox"],
    },
    "spooky_(ahegaokami)": {
        "character": ["spooky_(ahegaokami)"],
        "trigger": ["spooky \\(ahegaokami\\), east asian mythology"],
    },
    "draco_(draco)": {
        "character": ["draco_(draco)"],
        "trigger": ["draco \\(draco\\), pokemon"],
    },
    "yesenia_(character)": {
        "character": ["yesenia_(character)"],
        "trigger": ["yesenia \\(character\\), mythology"],
    },
    "aryn_(the_dogsmith)": {
        "character": ["aryn_(the_dogsmith)"],
        "trigger": ["aryn \\(the dogsmith\\), christmas"],
    },
    "owen_(amadose)": {
        "character": ["owen_(amadose)"],
        "trigger": ["owen \\(amadose\\), disney"],
    },
    "ria_(piilsud)": {
        "character": ["ria_(piilsud)"],
        "trigger": ["ria \\(piilsud\\), drawpile"],
    },
    "gally_(monsterbunny)": {
        "character": ["gally_(monsterbunny)"],
        "trigger": ["gally \\(monsterbunny\\), my little pony"],
    },
    "daycare_attendant_(fnaf)": {
        "character": ["daycare_attendant_(fnaf)"],
        "trigger": ["daycare attendant \\(fnaf\\), scottgames"],
    },
    "tsuneaki": {"character": ["tsuneaki"], "trigger": ["tsuneaki, lifewonders"]},
    "kypper_(alexvanarsdale)": {
        "character": ["kypper_(alexvanarsdale)"],
        "trigger": ["kypper \\(alexvanarsdale\\), legends of amora"],
    },
    "chibi_(chibitay)": {
        "character": ["chibi_(chibitay)"],
        "trigger": ["chibi \\(chibitay\\), nintendo"],
    },
    "ariem_(sonic)": {
        "character": ["ariem_(sonic)"],
        "trigger": ["ariem \\(sonic\\), sonic the hedgehog \\(series\\)"],
    },
    "mae_(paige)": {
        "character": ["mae_(paige)"],
        "trigger": ["mae \\(paige\\), halloween"],
    },
    "thumper_(disney)": {
        "character": ["thumper_(disney)"],
        "trigger": ["thumper \\(disney\\), disney"],
    },
    "chief_smirnov": {
        "character": ["chief_smirnov"],
        "trigger": ["chief smirnov, blacksad"],
    },
    "zephyr": {"character": ["zephyr"], "trigger": ["zephyr, mythology"]},
    "shadow_wolf": {
        "character": ["shadow_wolf"],
        "trigger": ["shadow wolf, mythology"],
    },
    "alexandra_williams": {
        "character": ["alexandra_williams"],
        "trigger": ["alexandra williams, christmas"],
    },
    "jenny_(slither)": {
        "character": ["jenny_(slither)"],
        "trigger": ["jenny \\(slither\\), slither"],
    },
    "bruma": {"character": ["bruma"], "trigger": ["bruma, el arca"]},
    "lady_weavile": {
        "character": ["lady_weavile"],
        "trigger": ["lady weavile, pokemon mystery dungeon"],
    },
    "trash_bandicoot": {
        "character": ["trash_bandicoot"],
        "trigger": ["trash bandicoot, crash bandicoot \\(series\\)"],
    },
    "amber_(femsubamber)": {
        "character": ["amber_(femsubamber)"],
        "trigger": ["amber \\(femsubamber\\), alien \\(franchise\\)"],
    },
    "marty_the_zebra": {
        "character": ["marty_the_zebra"],
        "trigger": ["marty the zebra, dreamworks"],
    },
    "comet_(reindeer)": {
        "character": ["comet_(reindeer)"],
        "trigger": ["comet \\(reindeer\\), christmas"],
    },
    "spiral_staircase": {
        "character": ["spiral_staircase"],
        "trigger": ["spiral staircase, thaine"],
    },
    "ma-san": {"character": ["ma-san"], "trigger": ["ma-san, parappa the rapper"]},
    "asmodeus_(character)": {
        "character": ["asmodeus_(character)"],
        "trigger": ["asmodeus \\(character\\), mythology"],
    },
    "kona": {"character": ["kona"], "trigger": ["kona, mythology"]},
    "annabelle_cow": {
        "character": ["annabelle_cow"],
        "trigger": ["annabelle cow, mythology"],
    },
    "vera_(iskra)": {
        "character": ["vera_(iskra)"],
        "trigger": ["vera \\(iskra\\), christmas"],
    },
    "screw_loose_(mlp)": {
        "character": ["screw_loose_(mlp)"],
        "trigger": ["screw loose \\(mlp\\), my little pony"],
    },
    "gwen_(zaggatar)": {
        "character": ["gwen_(zaggatar)"],
        "trigger": ["gwen \\(zaggatar\\), mythology"],
    },
    "sasha_phyronix": {
        "character": ["sasha_phyronix"],
        "trigger": ["sasha phyronix, sony corporation"],
    },
    "minette": {"character": ["minette"], "trigger": ["minette, skullgirls"]},
    "arh_(character)": {
        "character": ["arh_(character)"],
        "trigger": ["arh \\(character\\), mythology"],
    },
    "marjani_(character)": {
        "character": ["marjani_(character)"],
        "trigger": ["marjani \\(character\\), the shadow of light"],
    },
    "peko": {"character": ["peko"], "trigger": ["peko, doraemon"]},
    "taranza": {"character": ["taranza"], "trigger": ["taranza, kirby \\(series\\)"]},
    "rita_skopt": {
        "character": ["rita_skopt"],
        "trigger": ["rita skopt, warner brothers"],
    },
    "tybalt_(animal_crossing)": {
        "character": ["tybalt_(animal_crossing)"],
        "trigger": ["tybalt \\(animal crossing\\), animal crossing"],
    },
    "erakir": {"character": ["erakir"], "trigger": ["erakir, square enix"]},
    "claudette_dupri": {
        "character": ["claudette_dupri"],
        "trigger": ["claudette dupri, new looney tunes"],
    },
    "kurt_the_thunderfloof": {
        "character": ["kurt_the_thunderfloof"],
        "trigger": ["kurt the thunderfloof, mythology"],
    },
    "daedalus_vindryal": {
        "character": ["daedalus_vindryal"],
        "trigger": ["daedalus vindryal, mythology"],
    },
    "thrack": {"character": ["thrack"], "trigger": ["thrack, mythology"]},
    "noxus_poppy_(lol)": {
        "character": ["noxus_poppy_(lol)"],
        "trigger": ["noxus poppy \\(lol\\), riot games"],
    },
    "lief_woodcock": {
        "character": ["lief_woodcock"],
        "trigger": ["lief woodcock, mythology"],
    },
    "azul_alexander": {
        "character": ["azul_alexander"],
        "trigger": ["azul alexander, nintendo"],
    },
    "scp-2547-1": {
        "character": ["scp-2547-1"],
        "trigger": ["scp-2547-1, scp foundation"],
    },
    "ookami_(aggretsuko)": {
        "character": ["ookami_(aggretsuko)"],
        "trigger": ["ookami \\(aggretsuko\\), sanrio"],
    },
    "ose_(tas)": {
        "character": ["ose_(tas)"],
        "trigger": ["ose \\(tas\\), lifewonders"],
    },
    "cain_sentau": {
        "character": ["cain_sentau"],
        "trigger": ["cain sentau, mythology"],
    },
    "aruri": {"character": ["aruri"], "trigger": ["aruri, mythology"]},
    "garou_kazeoka": {
        "character": ["garou_kazeoka"],
        "trigger": ["garou kazeoka, nintendo switch"],
    },
    "trish_(invasormkiv)": {
        "character": ["trish_(invasormkiv)"],
        "trigger": ["trish \\(invasormkiv\\), calvin klein"],
    },
    "lyga": {"character": ["lyga"], "trigger": ["lyga, square enix"]},
    "monki": {"character": ["monki"], "trigger": ["monki, dragon ball"]},
    "wolfgang_(slickerwolf)": {
        "character": ["wolfgang_(slickerwolf)"],
        "trigger": ["wolfgang \\(slickerwolf\\), digimon"],
    },
    "airam": {"character": ["airam"], "trigger": ["airam, nintendo switch"]},
    "d0nk": {"character": ["d0nk"], "trigger": ["d0nk, warcraft"]},
    "alex_(happytimes)": {
        "character": ["alex_(happytimes)"],
        "trigger": ["alex \\(happytimes\\), avatar: the last airbender"],
    },
    "fenrir_osbone_(character)": {
        "character": ["fenrir_osbone_(character)"],
        "trigger": ["fenrir osbone \\(character\\), mythology"],
    },
    "fruit_slice_(yurusa)": {
        "character": ["fruit_slice_(yurusa)"],
        "trigger": ["fruit slice \\(yurusa\\), mythology"],
    },
    "tasha_lisets": {
        "character": ["tasha_lisets"],
        "trigger": ["tasha lisets, halloween"],
    },
    "katsukaka_(taokakaguy)": {
        "character": ["katsukaka_(taokakaguy)"],
        "trigger": ["katsukaka \\(taokakaguy\\), arc system works"],
    },
    "virgil_(maxydont)": {
        "character": ["virgil_(maxydont)"],
        "trigger": ["virgil \\(maxydont\\), mythology"],
    },
    "remy_dragon": {
        "character": ["remy_dragon"],
        "trigger": ["remy dragon, mythology"],
    },
    "beth_(redustheriotact)": {
        "character": ["beth_(redustheriotact)"],
        "trigger": ["beth \\(redustheriotact\\), christmas"],
    },
    "sam_(zoroark)": {
        "character": ["sam_(zoroark)"],
        "trigger": ["sam \\(zoroark\\), pokemon"],
    },
    "kc_(kingcreep105)": {
        "character": ["kc_(kingcreep105)"],
        "trigger": ["kc \\(kingcreep105\\), nintendo"],
    },
    "kineta": {"character": ["kineta"], "trigger": ["kineta, mythology"]},
    "alan_white": {
        "character": ["alan_white"],
        "trigger": ["alan white, year of the rabbit"],
    },
    "vivi_ornitier": {
        "character": ["vivi_ornitier"],
        "trigger": ["vivi ornitier, square enix"],
    },
    "neopatamonx": {"character": ["neopatamonx"], "trigger": ["neopatamonx, digimon"]},
    "isis_(nightfaux)": {
        "character": ["isis_(nightfaux)"],
        "trigger": ["isis \\(nightfaux\\), nintendo"],
    },
    "nohni_wabanda": {
        "character": ["nohni_wabanda"],
        "trigger": ["nohni wabanda, furafterdark"],
    },
    "buttercup_(powerpuff_girls)": {
        "character": ["buttercup_(powerpuff_girls)"],
        "trigger": ["buttercup \\(powerpuff girls\\), cartoon network"],
    },
    "abigail_roo": {
        "character": ["abigail_roo"],
        "trigger": ["abigail roo, mythology"],
    },
    "sasha_(sashabelle)": {
        "character": ["sasha_(sashabelle)"],
        "trigger": ["sasha \\(sashabelle\\), christmas"],
    },
    "seff_(seff)": {
        "character": ["seff_(seff)"],
        "trigger": ["seff \\(seff\\), camp pines"],
    },
    "hannibal_lecter": {
        "character": ["hannibal_lecter"],
        "trigger": ["hannibal lecter, hannibal \\(series\\)"],
    },
    "chipfox": {"character": ["chipfox"], "trigger": ["chipfox, mythology"]},
    "ferloo": {"character": ["ferloo"], "trigger": ["ferloo, mythology"]},
    "pocket_jabari": {
        "character": ["pocket_jabari"],
        "trigger": ["pocket jabari, nintendo"],
    },
    "kra-ra": {"character": ["kra-ra"], "trigger": ["kra-ra, criticalhit64"]},
    "lucia_traveyne": {
        "character": ["lucia_traveyne"],
        "trigger": ["lucia traveyne, mythology"],
    },
    "loike": {"character": ["loike"], "trigger": ["loike, mythology"]},
    "drudgegut": {"character": ["drudgegut"], "trigger": ["drudgegut, star fox"]},
    "woofy_(woofyrainshadow)": {
        "character": ["woofy_(woofyrainshadow)"],
        "trigger": ["woofy \\(woofyrainshadow\\), subscribestar"],
    },
    "avarice_panthera_leo": {
        "character": ["avarice_panthera_leo"],
        "trigger": ["avarice panthera leo, mythology"],
    },
    "sakyubasu": {"character": ["sakyubasu"], "trigger": ["sakyubasu, mythology"]},
    "genji_(overwatch)": {
        "character": ["genji_(overwatch)"],
        "trigger": ["genji \\(overwatch\\), overwatch"],
    },
    "katamra_(spazman)": {
        "character": ["katamra_(spazman)"],
        "trigger": ["katamra \\(spazman\\), mythology"],
    },
    "mango_(mangobird)": {
        "character": ["mango_(mangobird)"],
        "trigger": ["mango \\(mangobird\\), mythology"],
    },
    "saitama_(one-punch_man)": {
        "character": ["saitama_(one-punch_man)"],
        "trigger": ["saitama \\(one-punch man\\), one-punch man"],
    },
    "mouse_girl_(youki029)": {
        "character": ["mouse_girl_(youki029)"],
        "trigger": ["mouse girl \\(youki029\\), halloween"],
    },
    "miri_rodgers": {
        "character": ["miri_rodgers"],
        "trigger": ["miri rodgers, the wayward astronomer"],
    },
    "abaddon": {"character": ["abaddon"], "trigger": ["abaddon, pokemon"]},
    "mrs._amp_(mramp)": {
        "character": ["mrs._amp_(mramp)"],
        "trigger": ["mrs. amp \\(mramp\\), knuckle up!"],
    },
    "anklav": {"character": ["anklav"], "trigger": ["anklav, mythology"]},
    "elaz": {"character": ["elaz"], "trigger": ["elaz, mythology"]},
    "birddi": {"character": ["birddi"], "trigger": ["birddi, mythology"]},
    "kabbu_(bug_fables)": {
        "character": ["kabbu_(bug_fables)"],
        "trigger": ["kabbu \\(bug fables\\), bug fables"],
    },
    "sam_(orf)": {"character": ["sam_(orf)"], "trigger": ["sam \\(orf\\), halloween"]},
    "salomonkun": {"character": ["salomonkun"], "trigger": ["salomonkun, lifewonders"]},
    "ka_sarra": {"character": ["ka_sarra"], "trigger": ["ka sarra, age of empires"]},
    "starock": {"character": ["starock"], "trigger": ["starock, mythology"]},
    "avery_willard": {
        "character": ["avery_willard"],
        "trigger": ["avery willard, pokemon"],
    },
    "retrospecter_(character)": {
        "character": ["retrospecter_(character)"],
        "trigger": ["retrospecter \\(character\\), mythology"],
    },
    "kilix": {"character": ["kilix"], "trigger": ["kilix, invader zim"]},
    "nil": {"character": ["nil"], "trigger": ["nil, mythology"]},
    "funtime_chica_(fnaf)": {
        "character": ["funtime_chica_(fnaf)"],
        "trigger": ["funtime chica \\(fnaf\\), scottgames"],
    },
    "arizel": {"character": ["arizel"], "trigger": ["arizel, halloween"]},
    "voss_(beastars)": {
        "character": ["voss_(beastars)"],
        "trigger": ["voss \\(beastars\\), beastars"],
    },
    "annie_hole": {
        "character": ["annie_hole"],
        "trigger": ["annie hole, annie and the mirror goat"],
    },
    "lotion_cat_(kekitopu)": {
        "character": ["lotion_cat_(kekitopu)"],
        "trigger": ["lotion cat \\(kekitopu\\), pokemon"],
    },
    "swift_(sleekhusky)": {
        "character": ["swift_(sleekhusky)"],
        "trigger": ["swift \\(sleekhusky\\)"],
    },
    "alsander": {"character": ["alsander"], "trigger": ["alsander, pokemon"]},
    "tami_k_maru_(yourfavoritelemonade)": {
        "character": ["tami_k_maru_(yourfavoritelemonade)"],
        "trigger": ["tami k maru \\(yourfavoritelemonade\\), stories of the few"],
    },
    "usawa_fuwakaru": {
        "character": ["usawa_fuwakaru"],
        "trigger": ["usawa fuwakaru, nintendo"],
    },
    "sophring_hao": {
        "character": ["sophring_hao"],
        "trigger": ["sophring hao, full attack"],
    },
    "tony_(tonytoran)": {
        "character": ["tony_(tonytoran)"],
        "trigger": ["tony \\(tonytoran\\), nintendo"],
    },
    "pathfinder_(apex_legends)": {
        "character": ["pathfinder_(apex_legends)"],
        "trigger": ["pathfinder \\(apex legends\\), apex legends"],
    },
    "shirakami_fubuki": {
        "character": ["shirakami_fubuki"],
        "trigger": ["shirakami fubuki, hololive"],
    },
    "steban_(skailla)": {
        "character": ["steban_(skailla)"],
        "trigger": ["steban \\(skailla\\), no north"],
    },
    "sin_the_hedgehog": {
        "character": ["sin_the_hedgehog"],
        "trigger": ["sin the hedgehog, sonic the hedgehog \\(series\\)"],
    },
    "mikey_(mikeyuk)": {
        "character": ["mikey_(mikeyuk)"],
        "trigger": ["mikey \\(mikeyuk\\), stone guardians"],
    },
    "zeta_(zeraora)": {
        "character": ["zeta_(zeraora)"],
        "trigger": ["zeta \\(zeraora\\), pokemon"],
    },
    "amber_puppy": {
        "character": ["amber_puppy"],
        "trigger": ["amber puppy, halloween"],
    },
    "zach_(zer0rebel4)": {
        "character": ["zach_(zer0rebel4)"],
        "trigger": ["zach \\(zer0rebel4\\), patreon"],
    },
    "admiral_brickell": {
        "character": ["admiral_brickell"],
        "trigger": ["admiral brickell, ninja kiwi"],
    },
    "maxine_boulevard": {
        "character": ["maxine_boulevard"],
        "trigger": ["maxine boulevard, disney"],
    },
    "shanher_(character)": {
        "character": ["shanher_(character)"],
        "trigger": ["shanher \\(character\\), mythology"],
    },
    "mika_(lunarpanda8686)": {
        "character": ["mika_(lunarpanda8686)"],
        "trigger": ["mika \\(lunarpanda8686\\), mr. osomatsu"],
    },
    "wynter_(fr34kpet)": {
        "character": ["wynter_(fr34kpet)"],
        "trigger": ["wynter \\(fr34kpet\\), 4lung"],
    },
    "obsidius": {"character": ["obsidius"], "trigger": ["obsidius, lifewonders"]},
    "mivliano_10-c": {"character": ["mivliano_10-c"], "trigger": ["mivliano 10-c"]},
    "rio_(miu)": {
        "character": ["rio_(miu)"],
        "trigger": ["rio \\(miu\\), clubstripes"],
    },
    "meeya": {"character": ["meeya"], "trigger": ["meeya, rpg densetsu hepoi"]},
    "circe": {"character": ["circe"], "trigger": ["circe, mythology"]},
    "koul_fardreamer": {
        "character": ["koul_fardreamer"],
        "trigger": ["koul fardreamer, s.t.a.l.k.e.r."],
    },
    "dard": {"character": ["dard"], "trigger": ["dard, drakuun"]},
    "lucky_(101_dalmatians)": {
        "character": ["lucky_(101_dalmatians)"],
        "trigger": ["lucky \\(101 dalmatians\\), disney"],
    },
    "traveler": {
        "character": ["traveler"],
        "trigger": ["traveler, journey \\(game\\)"],
    },
    "protagonist_(left_4_dead)": {
        "character": ["protagonist_(left_4_dead)"],
        "trigger": ["protagonist \\(left 4 dead\\), left 4 dead \\(series\\)"],
    },
    "alister_azimuth": {
        "character": ["alister_azimuth"],
        "trigger": ["alister azimuth, sony corporation"],
    },
    "ultra_nyan": {
        "character": ["ultra_nyan"],
        "trigger": ["ultra nyan, ultraman \\(series\\)"],
    },
    "lully_pop": {"character": ["lully_pop"], "trigger": ["lully pop, mythology"]},
    "gor_(tomcat)": {
        "character": ["gor_(tomcat)"],
        "trigger": ["gor \\(tomcat\\), chippendales"],
    },
    "elite_four": {"character": ["elite_four"], "trigger": ["elite four, pokemon"]},
    "airachnid": {"character": ["airachnid"], "trigger": ["airachnid, takara tomy"]},
    "scott_(fasttrack37d)": {
        "character": ["scott_(fasttrack37d)"],
        "trigger": ["scott \\(fasttrack37d\\), mythology"],
    },
    "thomas_(regular_show)": {
        "character": ["thomas_(regular_show)"],
        "trigger": ["thomas \\(regular show\\), cartoon network"],
    },
    "gracie_(animal_crossing)": {
        "character": ["gracie_(animal_crossing)"],
        "trigger": ["gracie \\(animal crossing\\), animal crossing"],
    },
    "glitch_(the_gamercat)": {
        "character": ["glitch_(the_gamercat)"],
        "trigger": ["glitch \\(the gamercat\\), the gamercat"],
    },
    "snowy_(yuki-the-fox)": {
        "character": ["snowy_(yuki-the-fox)"],
        "trigger": ["snowy \\(yuki-the-fox\\), yuki-the-fox"],
    },
    "fili-second_(mlp)": {
        "character": ["fili-second_(mlp)"],
        "trigger": ["fili-second \\(mlp\\), my little pony"],
    },
    "stacey_(goof_troop)": {
        "character": ["stacey_(goof_troop)"],
        "trigger": ["stacey \\(goof troop\\), disney"],
    },
    "mimi_(paper_mario)": {
        "character": ["mimi_(paper_mario)"],
        "trigger": ["mimi \\(paper mario\\), mario bros"],
    },
    "shen_(archshen)": {
        "character": ["shen_(archshen)"],
        "trigger": ["shen \\(archshen\\), bethesda softworks"],
    },
    "calem_(ruddrbtt)": {
        "character": ["calem_(ruddrbtt)"],
        "trigger": ["calem \\(ruddrbtt\\), mythology"],
    },
    "minoru_mineta": {
        "character": ["minoru_mineta"],
        "trigger": ["minoru mineta, my hero academia"],
    },
    "redmond": {"character": ["redmond"], "trigger": ["redmond, whiplash \\(game\\)"]},
    "horsie": {"character": ["horsie"], "trigger": ["horsie, warcraft"]},
    "mech_(mechedragon)": {
        "character": ["mech_(mechedragon)"],
        "trigger": ["mech \\(mechedragon\\), mythology"],
    },
    "sueli_(joaoppereiraus)": {
        "character": ["sueli_(joaoppereiraus)"],
        "trigger": ["sueli \\(joaoppereiraus\\), jonas brasileiro \\(copyright\\)"],
    },
    "undyne_the_undying": {
        "character": ["undyne_the_undying"],
        "trigger": ["undyne the undying, undertale \\(series\\)"],
    },
    "victoria_violeta_(usuario2)": {
        "character": ["victoria_violeta_(usuario2)"],
        "trigger": ["victoria violeta \\(usuario2\\), christmas"],
    },
    "isabelle_wilde": {
        "character": ["isabelle_wilde"],
        "trigger": ["isabelle wilde, disney"],
    },
    "osiris_henschel": {
        "character": ["osiris_henschel"],
        "trigger": ["osiris henschel, christmas"],
    },
    "hibari": {"character": ["hibari"], "trigger": ["hibari, pokemon"]},
    "annie_(disfigure/rafielo)": {
        "character": ["annie_(disfigure/rafielo)"],
        "trigger": ["annie \\(disfigure/rafielo\\), star wars"],
    },
    "tyroo_(character)": {
        "character": ["tyroo_(character)"],
        "trigger": ["tyroo \\(character\\), switcher-roo"],
    },
    "captain_flintlock_(felino)": {
        "character": ["captain_flintlock_(felino)"],
        "trigger": ["captain flintlock \\(felino\\), magical castle"],
    },
    "stan_borowski": {
        "character": ["stan_borowski"],
        "trigger": ["stan borowski, night in the woods"],
    },
    "karhyena": {"character": ["karhyena"], "trigger": ["karhyena, pokemon"]},
    "jamie_(boosterpang)": {
        "character": ["jamie_(boosterpang)"],
        "trigger": ["jamie \\(boosterpang\\), mythology"],
    },
    "wonder_boy": {"character": ["wonder_boy"], "trigger": ["wonder boy, sega"]},
    "vivian_(animal_crossing)": {
        "character": ["vivian_(animal_crossing)"],
        "trigger": ["vivian \\(animal crossing\\), animal crossing"],
    },
    "ereki_kagami": {
        "character": ["ereki_kagami"],
        "trigger": ["ereki kagami, halloween"],
    },
    "serona_shea": {
        "character": ["serona_shea"],
        "trigger": ["serona shea, mythology"],
    },
    "ithilwen_galanodel": {
        "character": ["ithilwen_galanodel"],
        "trigger": ["ithilwen galanodel, mythology"],
    },
    "jackie_demon": {
        "character": ["jackie_demon"],
        "trigger": ["jackie demon, mythology"],
    },
    "thorin": {"character": ["thorin"], "trigger": ["thorin, mythology"]},
    "strelizia": {"character": ["strelizia"], "trigger": ["strelizia, studio trigger"]},
    "dobie_(animal_crossing)": {
        "character": ["dobie_(animal_crossing)"],
        "trigger": ["dobie \\(animal crossing\\), animal crossing"],
    },
    "indie_(xanderblaze)": {
        "character": ["indie_(xanderblaze)"],
        "trigger": ["indie \\(xanderblaze\\), helluva boss"],
    },
    "annie_hill": {
        "character": ["annie_hill"],
        "trigger": ["annie hill, t.u.f.f. puppy"],
    },
    "pebble_(letodoesart)": {
        "character": ["pebble_(letodoesart)"],
        "trigger": ["pebble \\(letodoesart\\), christmas"],
    },
    "daphniir": {"character": ["daphniir"], "trigger": ["daphniir, mythology"]},
    "manager_(gym_pals)": {
        "character": ["manager_(gym_pals)"],
        "trigger": ["manager \\(gym pals\\), gym pals"],
    },
    "cassius_(adastra)": {
        "character": ["cassius_(adastra)"],
        "trigger": ["cassius \\(adastra\\), adastra \\(series\\)"],
    },
    "malina_(helltaker)": {
        "character": ["malina_(helltaker)"],
        "trigger": ["malina \\(helltaker\\), helltaker"],
    },
    "acrystra": {"character": ["acrystra"], "trigger": ["acrystra, halloween"]},
    "dumderg's_signature_cutie": {
        "character": ["dumderg's_signature_cutie"],
        "trigger": ["dumderg's signature cutie, mythology"],
    },
    "arlo_(amazingcanislupus)": {
        "character": ["arlo_(amazingcanislupus)"],
        "trigger": ["arlo \\(amazingcanislupus\\), mythology"],
    },
    "stoat_(inscryption)": {
        "character": ["stoat_(inscryption)"],
        "trigger": ["stoat \\(inscryption\\), inscryption"],
    },
    "noah_(codymathews)": {
        "character": ["noah_(codymathews)"],
        "trigger": ["noah \\(codymathews\\), catchingup"],
    },
    "shalya_bushtail": {
        "character": ["shalya_bushtail"],
        "trigger": ["shalya bushtail, christmas"],
    },
    "viscunam": {"character": ["viscunam"], "trigger": ["viscunam, lifewonders"]},
    "agro_antirrhopus_(character)": {
        "character": ["agro_antirrhopus_(character)"],
        "trigger": ["agro antirrhopus \\(character\\), mythology"],
    },
    "takeru_takaishi": {
        "character": ["takeru_takaishi"],
        "trigger": ["takeru takaishi, digimon"],
    },
    "huska": {"character": ["huska"], "trigger": ["huska, love can be different"]},
    "huckleberry_hound": {
        "character": ["huckleberry_hound"],
        "trigger": ["huckleberry hound, the huckleberry hound show"],
    },
    "stella_(over_the_hedge)": {
        "character": ["stella_(over_the_hedge)"],
        "trigger": ["stella \\(over the hedge\\), over the hedge"],
    },
    "trico_(the_last_guardian)": {
        "character": ["trico_(the_last_guardian)"],
        "trigger": ["trico \\(the last guardian\\), the last guardian"],
    },
    "blooregard": {
        "character": ["blooregard"],
        "trigger": ["blooregard, cartoon network"],
    },
    "skeletor": {"character": ["skeletor"], "trigger": ["skeletor, mattel"]},
    "philip_j._fry": {
        "character": ["philip_j._fry"],
        "trigger": ["philip j. fry, comedy central"],
    },
    "rey_(satsukii)": {
        "character": ["rey_(satsukii)"],
        "trigger": ["rey \\(satsukii\\), mythology"],
    },
    "frag_(furfragged)": {
        "character": ["frag_(furfragged)"],
        "trigger": ["frag \\(furfragged\\), christmas"],
    },
    "officer_jenny": {
        "character": ["officer_jenny"],
        "trigger": ["officer jenny, pokemon"],
    },
    "allin": {"character": ["allin"], "trigger": ["allin, my little pony"]},
    "sofiya_ivanova": {
        "character": ["sofiya_ivanova"],
        "trigger": ["sofiya ivanova, pokemon"],
    },
    "cassiopeia_(lol)": {
        "character": ["cassiopeia_(lol)"],
        "trigger": ["cassiopeia \\(lol\\), riot games"],
    },
    "kira_redpaw": {
        "character": ["kira_redpaw"],
        "trigger": ["kira redpaw, mythology"],
    },
    "beach_ball_(character)": {
        "character": ["beach_ball_(character)"],
        "trigger": ["beach ball \\(character\\), my little pony"],
    },
    "gunnar": {"character": ["gunnar"], "trigger": ["gunnar, incestaroos"]},
    "coop_burtonburger": {
        "character": ["coop_burtonburger"],
        "trigger": ["coop burtonburger, kid vs. kat"],
    },
    "deku_link": {
        "character": ["deku_link"],
        "trigger": ["deku link, the legend of zelda"],
    },
    "tom_(lm)": {"character": ["tom_(lm)"], "trigger": ["tom \\(lm\\), love mechanic"]},
    "maxtheshadowdragon": {
        "character": ["maxtheshadowdragon"],
        "trigger": ["maxtheshadowdragon, mythology"],
    },
    "rorick_kintana": {
        "character": ["rorick_kintana"],
        "trigger": ["rorick kintana, european mythology"],
    },
    "ruby_blossom": {
        "character": ["ruby_blossom"],
        "trigger": ["ruby blossom, my little pony"],
    },
    "excalibur_(warframe)": {
        "character": ["excalibur_(warframe)"],
        "trigger": ["excalibur \\(warframe\\), warframe"],
    },
    "okami_wolf": {"character": ["okami_wolf"], "trigger": ["okami wolf, mythology"]},
    "tam_(tamfox)": {
        "character": ["tam_(tamfox)"],
        "trigger": ["tam \\(tamfox\\), draco32588"],
    },
    "lance_(radiantblaze)": {
        "character": ["lance_(radiantblaze)"],
        "trigger": ["lance \\(radiantblaze\\)"],
    },
    "gelbun": {"character": ["gelbun"], "trigger": ["gelbun, christmas"]},
    "eeple": {"character": ["eeple"], "trigger": ["eeple"]},
    "liberty_(bluecoffeedog)": {
        "character": ["liberty_(bluecoffeedog)"],
        "trigger": ["liberty \\(bluecoffeedog\\), my little pony"],
    },
    "bunga": {"character": ["bunga"], "trigger": ["bunga, disney"]},
    "nanomoochines": {
        "character": ["nanomoochines"],
        "trigger": ["nanomoochines, royal small arms factory"],
    },
    "iva_(cornchip21)": {
        "character": ["iva_(cornchip21)"],
        "trigger": ["iva \\(cornchip21\\), pokemon"],
    },
    "megumin_(konosuba)": {
        "character": ["megumin_(konosuba)"],
        "trigger": [
            "megumin \\(konosuba\\), konosuba: god's blessing on this wonderful world!"
        ],
    },
    "arthur_(lapinbeau)": {
        "character": ["arthur_(lapinbeau)"],
        "trigger": ["arthur \\(lapinbeau\\), easter"],
    },
    "jackie_hopps_(grummancat)": {
        "character": ["jackie_hopps_(grummancat)"],
        "trigger": ["jackie hopps \\(grummancat\\), disney"],
    },
    "bramble_(hitsunekun)": {
        "character": ["bramble_(hitsunekun)"],
        "trigger": ["bramble \\(hitsunekun\\), square enix"],
    },
    "adhara": {"character": ["adhara"], "trigger": ["adhara, mythology"]},
    "alex_(minecraft)": {
        "character": ["alex_(minecraft)"],
        "trigger": ["alex \\(minecraft\\), minecraft"],
    },
    "kitsuneinu": {"character": ["kitsuneinu"], "trigger": ["kitsuneinu, mythology"]},
    "major_wolf": {"character": ["major_wolf"], "trigger": ["major wolf, mythology"]},
    "quse": {"character": ["quse"], "trigger": ["quse, nintendo"]},
    "sol_the_guilmon": {
        "character": ["sol_the_guilmon"],
        "trigger": ["sol the guilmon, digimon"],
    },
    "adaline_(sharemyshipment)": {
        "character": ["adaline_(sharemyshipment)"],
        "trigger": ["adaline \\(sharemyshipment\\), my little pony"],
    },
    "silver_fox_(kemono_friends)": {
        "character": ["silver_fox_(kemono_friends)"],
        "trigger": ["silver fox \\(kemono friends\\), kemono friends"],
    },
    "lo": {"character": ["lo"], "trigger": ["lo, disney"]},
    "thaylen": {"character": ["thaylen"], "trigger": ["thaylen, mythology"]},
    "lanya_(shian)": {
        "character": ["lanya_(shian)"],
        "trigger": ["lanya \\(shian\\), mythology"],
    },
    "zahk_(knight)": {
        "character": ["zahk_(knight)"],
        "trigger": ["zahk \\(knight\\), my little pony"],
    },
    "queen_rain_shine_(mlp)": {
        "character": ["queen_rain_shine_(mlp)"],
        "trigger": ["queen rain shine \\(mlp\\), my little pony"],
    },
    "maria_(wffl)": {
        "character": ["maria_(wffl)"],
        "trigger": ["maria \\(wffl\\), mythology"],
    },
    "angus_(critterclaws)": {
        "character": ["angus_(critterclaws)"],
        "trigger": ["angus \\(critterclaws\\), mythology"],
    },
    "ol'_blue": {"character": ["ol'_blue"], "trigger": ["ol' blue, cartoon network"]},
    "brooz_(interspecies_reviewers)": {
        "character": ["brooz_(interspecies_reviewers)"],
        "trigger": ["brooz \\(interspecies reviewers\\), interspecies reviewers"],
    },
    "blue-wolfy": {"character": ["blue-wolfy"], "trigger": ["blue-wolfy, nintendo"]},
    "rio_(botter_dork)": {
        "character": ["rio_(botter_dork)"],
        "trigger": ["rio \\(botter dork\\), no nut november"],
    },
    "alice_(kairaanix)": {
        "character": ["alice_(kairaanix)"],
        "trigger": ["alice \\(kairaanix\\), mortal kombat"],
    },
    "rui_nikaido_(odd_taxi)": {
        "character": ["rui_nikaido_(odd_taxi)"],
        "trigger": ["rui nikaido \\(odd taxi\\), odd taxi"],
    },
    "funny_gay_rat": {
        "character": ["funny_gay_rat"],
        "trigger": ["funny gay rat, blender \\(software\\)"],
    },
    "renamon_(dogzeela)": {
        "character": ["renamon_(dogzeela)"],
        "trigger": ["renamon \\(dogzeela\\), digimon"],
    },
    "josephine_rodgers": {
        "character": ["josephine_rodgers"],
        "trigger": ["josephine rodgers, cats n' cameras"],
    },
    "anthro_anon": {"character": ["anthro_anon"], "trigger": ["anthro anon, disney"]},
    "fox_(skunk_fu)": {
        "character": ["fox_(skunk_fu)"],
        "trigger": ["fox \\(skunk fu\\), skunk fu"],
    },
    "tom_(rq)": {"character": ["tom_(rq)"], "trigger": ["tom \\(rq\\), ruby quest"]},
    "mitzi_(animal_crossing)": {
        "character": ["mitzi_(animal_crossing)"],
        "trigger": ["mitzi \\(animal crossing\\), animal crossing"],
    },
    "milkie_souris": {
        "character": ["milkie_souris"],
        "trigger": ["milkie souris, milkjunkie"],
    },
    "mayor_pauline": {
        "character": ["mayor_pauline"],
        "trigger": ["mayor pauline, mario bros"],
    },
    "pashmina_(hamtaro)": {
        "character": ["pashmina_(hamtaro)"],
        "trigger": ["pashmina \\(hamtaro\\), hamtaro \\(series\\)"],
    },
    "wazumi": {"character": ["wazumi"], "trigger": ["wazumi, mythology"]},
    "megatron": {"character": ["megatron"], "trigger": ["megatron, takara tomy"]},
    "sandy_(hamtaro)": {
        "character": ["sandy_(hamtaro)"],
        "trigger": ["sandy \\(hamtaro\\), hamtaro \\(series\\)"],
    },
    "bailey_(os)": {
        "character": ["bailey_(os)"],
        "trigger": ["bailey \\(os\\), mythology"],
    },
    "calheb_(calheb-db)": {
        "character": ["calheb_(calheb-db)"],
        "trigger": ["calheb \\(calheb-db\\), mythology"],
    },
    "kooper": {"character": ["kooper"], "trigger": ["kooper, mario bros"]},
    "glowfox_(character)": {
        "character": ["glowfox_(character)"],
        "trigger": ["glowfox \\(character\\), mythology"],
    },
    "boo-boo_bear": {
        "character": ["boo-boo_bear"],
        "trigger": ["boo-boo bear, yogi bear"],
    },
    "moto_moto": {"character": ["moto_moto"], "trigger": ["moto moto, dreamworks"]},
    "petunia_pig": {
        "character": ["petunia_pig"],
        "trigger": ["petunia pig, warner brothers"],
    },
    "benson_dunwoody": {
        "character": ["benson_dunwoody"],
        "trigger": ["benson dunwoody, cartoon network"],
    },
    "padme_amidala": {
        "character": ["padme_amidala"],
        "trigger": ["padme amidala, star wars"],
    },
    "geralt_of_rivia": {
        "character": ["geralt_of_rivia"],
        "trigger": ["geralt of rivia, the witcher"],
    },
    "junip": {"character": ["junip"], "trigger": ["junip, mythology"]},
    "bailey_(housepets!)": {
        "character": ["bailey_(housepets!)"],
        "trigger": ["bailey \\(housepets!\\), housepets!"],
    },
    "rapunzel_(disney)": {
        "character": ["rapunzel_(disney)"],
        "trigger": ["rapunzel \\(disney\\), disney"],
    },
    "topaz_(lipton)": {
        "character": ["topaz_(lipton)"],
        "trigger": ["topaz \\(lipton\\), dc comics"],
    },
    "mocha_softpaw": {
        "character": ["mocha_softpaw"],
        "trigger": ["mocha softpaw, mochasp"],
    },
    "wing_wu": {"character": ["wing_wu"], "trigger": ["wing wu, kung fu panda"]},
    "tanis_(ghoul_school)": {
        "character": ["tanis_(ghoul_school)"],
        "trigger": ["tanis \\(ghoul school\\), ghoul school"],
    },
    "dan_(smarticus)": {
        "character": ["dan_(smarticus)"],
        "trigger": ["dan \\(smarticus\\), bad dragon"],
    },
    "yu_narukami": {"character": ["yu_narukami"], "trigger": ["yu narukami, sega"]},
    "johnny_worthington": {
        "character": ["johnny_worthington"],
        "trigger": ["johnny worthington, disney"],
    },
    "saddle_rager_(mlp)": {
        "character": ["saddle_rager_(mlp)"],
        "trigger": ["saddle rager \\(mlp\\), my little pony"],
    },
    "rick_(dream_and_nightmare)": {
        "character": ["rick_(dream_and_nightmare)"],
        "trigger": ["rick \\(dream and nightmare\\), mythology"],
    },
    "kane_(kabscorner)": {
        "character": ["kane_(kabscorner)"],
        "trigger": ["kane \\(kabscorner\\), mythology"],
    },
    "anubis_(lollipopcon)": {
        "character": ["anubis_(lollipopcon)"],
        "trigger": ["anubis \\(lollipopcon\\), middle eastern mythology"],
    },
    "nic_(dewott)": {
        "character": ["nic_(dewott)"],
        "trigger": ["nic \\(dewott\\), pokemon"],
    },
    "russie": {"character": ["russie"], "trigger": ["russie, mythology"]},
    "victoria_(two-ts)": {
        "character": ["victoria_(two-ts)"],
        "trigger": ["victoria \\(two-ts\\), my life with fel"],
    },
    "lootz": {"character": ["lootz"], "trigger": ["lootz, mythology"]},
    "mogmaw": {"character": ["mogmaw"], "trigger": ["mogmaw, las lindas"]},
    "jinjing-yu": {"character": ["jinjing-yu"], "trigger": ["jinjing-yu, patreon"]},
    "empress_jasana": {
        "character": ["empress_jasana"],
        "trigger": ["empress jasana, master of orion"],
    },
    "patrick_(david_siegl)": {
        "character": ["patrick_(david_siegl)"],
        "trigger": ["patrick \\(david siegl\\)"],
    },
    "flo_(overflo207)": {
        "character": ["flo_(overflo207)"],
        "trigger": ["flo \\(overflo207\\), nintendo"],
    },
    "zera_stormfire": {
        "character": ["zera_stormfire"],
        "trigger": ["zera stormfire, mythology"],
    },
    "myri_(fvt)": {
        "character": ["myri_(fvt)"],
        "trigger": ["myri \\(fvt\\), fairies vs tentacles"],
    },
    "ms._moona": {"character": ["ms._moona"], "trigger": ["ms. moona, pokemon"]},
    "walrider_(outlast)": {
        "character": ["walrider_(outlast)"],
        "trigger": ["walrider \\(outlast\\), outlast"],
    },
    "altavy_(altavy)": {
        "character": ["altavy_(altavy)"],
        "trigger": ["altavy \\(altavy\\), bunny and fox world"],
    },
    "kej_(kejifox)": {
        "character": ["kej_(kejifox)"],
        "trigger": ["kej \\(kejifox\\), my little pony"],
    },
    "jaiden_animations": {
        "character": ["jaiden_animations"],
        "trigger": ["jaiden animations, youtube"],
    },
    "tracy_siren": {"character": ["tracy_siren"], "trigger": ["tracy siren, disney"]},
    "ritsuka_fujimaru": {
        "character": ["ritsuka_fujimaru"],
        "trigger": ["ritsuka fujimaru, type-moon"],
    },
    "aiko_(infamousrel)": {
        "character": ["aiko_(infamousrel)"],
        "trigger": ["aiko \\(infamousrel\\), square enix"],
    },
    "kaban-chan": {
        "character": ["kaban-chan"],
        "trigger": ["kaban-chan, kemono friends"],
    },
    "granbun": {"character": ["granbun"], "trigger": ["granbun, christmas"]},
    "chili_(zummeng)": {
        "character": ["chili_(zummeng)"],
        "trigger": ["chili \\(zummeng\\), mythology"],
    },
    "kate_(playkids)": {
        "character": ["kate_(playkids)"],
        "trigger": ["kate \\(playkids\\), playkids"],
    },
    "mouritzeen_(lenyavok)": {
        "character": ["mouritzeen_(lenyavok)"],
        "trigger": ["mouritzeen \\(lenyavok\\), christmas"],
    },
    "professor_rena": {
        "character": ["professor_rena"],
        "trigger": ["professor rena, digimon"],
    },
    "michelle_catty": {
        "character": ["michelle_catty"],
        "trigger": ["michelle catty, pokemon"],
    },
    "shai_dreamcast": {
        "character": ["shai_dreamcast"],
        "trigger": ["shai dreamcast, nintendo"],
    },
    "mr._henderson": {
        "character": ["mr._henderson"],
        "trigger": ["mr. henderson, pokemon"],
    },
    "didi_(karakylia)": {
        "character": ["didi_(karakylia)"],
        "trigger": ["didi \\(karakylia\\), mythology"],
    },
    "max_(sci)": {
        "character": ["max_(sci)"],
        "trigger": ["max \\(sci\\), cartoon network"],
    },
    "rovik_(rovik1174)": {
        "character": ["rovik_(rovik1174)"],
        "trigger": ["rovik \\(rovik1174\\), mythology"],
    },
    "biscuit_(dashboom)": {
        "character": ["biscuit_(dashboom)"],
        "trigger": ["biscuit \\(dashboom\\), pokemon"],
    },
    "temmie_(deltarune)": {
        "character": ["temmie_(deltarune)"],
        "trigger": ["temmie \\(deltarune\\), undertale \\(series\\)"],
    },
    "mittens_(panapoliz)": {
        "character": ["mittens_(panapoliz)"],
        "trigger": ["mittens \\(panapoliz\\), pokemon"],
    },
    "sheelahi": {"character": ["sheelahi"], "trigger": ["sheelahi, the elder scrolls"]},
    "oro_(ungoliant)": {
        "character": ["oro_(ungoliant)"],
        "trigger": ["oro \\(ungoliant\\), nintendo"],
    },
    "doctor_starline": {
        "character": ["doctor_starline"],
        "trigger": ["doctor starline, sonic the hedgehog \\(series\\)"],
    },
    "noisy_(redace83)": {
        "character": ["noisy_(redace83)"],
        "trigger": ["noisy \\(redace83\\), system shock"],
    },
    "ryndion": {"character": ["ryndion"], "trigger": ["ryndion, mythology"]},
    "kermo_(kamui_shirow)": {
        "character": ["kermo_(kamui_shirow)"],
        "trigger": ["kermo \\(kamui shirow\\), delivery bear service"],
    },
    "ailin_gardevoir": {
        "character": ["ailin_gardevoir"],
        "trigger": ["ailin gardevoir, pokemon"],
    },
    "aloe_(interspecies_reviewers)": {
        "character": ["aloe_(interspecies_reviewers)"],
        "trigger": ["aloe \\(interspecies reviewers\\), interspecies reviewers"],
    },
    "dom_(animal_crossing)": {
        "character": ["dom_(animal_crossing)"],
        "trigger": ["dom \\(animal crossing\\), animal crossing"],
    },
    "aaron_(falcon_mccooper)": {
        "character": ["aaron_(falcon_mccooper)"],
        "trigger": ["aaron \\(falcon mccooper\\), patreon"],
    },
    "daniel_(pukaa)": {
        "character": ["daniel_(pukaa)"],
        "trigger": ["daniel \\(pukaa\\), disney"],
    },
    "cat_operator": {
        "character": ["cat_operator"],
        "trigger": ["cat operator, lifewonders"],
    },
    "tracy_(anaugi)": {
        "character": ["tracy_(anaugi)"],
        "trigger": ["tracy \\(anaugi\\), capcom"],
    },
    "doodle_(doodledoggy)": {
        "character": ["doodle_(doodledoggy)"],
        "trigger": ["doodle \\(doodledoggy\\), star trek"],
    },
    "ima_(imabunbun)": {
        "character": ["ima_(imabunbun)"],
        "trigger": ["ima \\(imabunbun\\), mythology"],
    },
    "danzer_(reptilligator)": {
        "character": ["danzer_(reptilligator)"],
        "trigger": ["danzer \\(reptilligator\\), mythology"],
    },
    "manna-mint": {"character": ["manna-mint"], "trigger": ["manna-mint, mythology"]},
    "tatum_koenig": {
        "character": ["tatum_koenig"],
        "trigger": ["tatum koenig, mythology"],
    },
    "bridgette_(thighlordash)": {
        "character": ["bridgette_(thighlordash)"],
        "trigger": ["bridgette \\(thighlordash\\), my little pony"],
    },
    "ratau_(cult_of_the_lamb)": {
        "character": ["ratau_(cult_of_the_lamb)"],
        "trigger": ["ratau \\(cult of the lamb\\), cult of the lamb"],
    },
    "vapula_(tas)": {
        "character": ["vapula_(tas)"],
        "trigger": ["vapula \\(tas\\), lifewonders"],
    },
    "tails_nine": {
        "character": ["tails_nine"],
        "trigger": ["tails nine, sonic the hedgehog \\(series\\)"],
    },
    "aaron_steele": {
        "character": ["aaron_steele"],
        "trigger": ["aaron steele, divine acid"],
    },
    "craftycorn": {
        "character": ["craftycorn"],
        "trigger": ["craftycorn, poppy playtime"],
    },
    "lucy_black": {"character": ["lucy_black"], "trigger": ["lucy black, better days"]},
    "watson_(sherlock_hound)": {
        "character": ["watson_(sherlock_hound)"],
        "trigger": ["watson \\(sherlock hound\\), sherlock hound \\(series\\)"],
    },
    "foshu_(character)": {
        "character": ["foshu_(character)"],
        "trigger": ["foshu \\(character\\), mythology"],
    },
    "jaque_smith": {
        "character": ["jaque_smith"],
        "trigger": ["jaque smith, brave new world \\(style wager\\)"],
    },
    "clarisse_(sabrina_online)": {
        "character": ["clarisse_(sabrina_online)"],
        "trigger": ["clarisse \\(sabrina online\\), sabrina online"],
    },
    "annie_(brian_mcpherson)": {
        "character": ["annie_(brian_mcpherson)"],
        "trigger": ["annie \\(brian mcpherson\\), halloween"],
    },
    "shin_(las_lindas)": {
        "character": ["shin_(las_lindas)"],
        "trigger": ["shin \\(las lindas\\), las lindas"],
    },
    "duke_(bad_dragon)": {
        "character": ["duke_(bad_dragon)"],
        "trigger": ["duke \\(bad dragon\\), bad dragon"],
    },
    "marth": {"character": ["marth"], "trigger": ["marth, nintendo"]},
    "anput": {"character": ["anput"], "trigger": ["anput, middle eastern mythology"]},
    "kraid": {"character": ["kraid"], "trigger": ["kraid, nintendo"]},
    "mojo_jojo": {
        "character": ["mojo_jojo"],
        "trigger": ["mojo jojo, cartoon network"],
    },
    "buck_(ice_age)": {
        "character": ["buck_(ice_age)"],
        "trigger": ["buck \\(ice age\\), ice age \\(series\\)"],
    },
    "guin": {"character": ["guin"], "trigger": ["guin, guin saga"]},
    "muffy_crosswire": {
        "character": ["muffy_crosswire"],
        "trigger": ["muffy crosswire, arthur \\(series\\)"],
    },
    "kanaya_maryam": {
        "character": ["kanaya_maryam"],
        "trigger": ["kanaya maryam, homestuck"],
    },
    "fancy_(fancy-fancy)": {
        "character": ["fancy_(fancy-fancy)"],
        "trigger": ["fancy \\(fancy-fancy\\), fancy-fancy \\(copyright\\)"],
    },
    "willow_(glopossum)": {
        "character": ["willow_(glopossum)"],
        "trigger": ["willow \\(glopossum\\), mythology"],
    },
    "eloise_(animal_crossing)": {
        "character": ["eloise_(animal_crossing)"],
        "trigger": ["eloise \\(animal crossing\\), animal crossing"],
    },
    "deku_princess": {
        "character": ["deku_princess"],
        "trigger": ["deku princess, majora's mask"],
    },
    "cayenne_(freckles)": {
        "character": ["cayenne_(freckles)"],
        "trigger": ["cayenne \\(freckles\\), sergeantbuck"],
    },
    "volteer": {"character": ["volteer"], "trigger": ["volteer, mythology"]},
    "acchan_(arkaid)": {
        "character": ["acchan_(arkaid)"],
        "trigger": ["acchan \\(arkaid\\), the neon children"],
    },
    "haley_(elysian_tail)": {
        "character": ["haley_(elysian_tail)"],
        "trigger": ["haley \\(elysian tail\\), dust: an elysian tail"],
    },
    "beruca_(glopossum)": {
        "character": ["beruca_(glopossum)"],
        "trigger": ["beruca \\(glopossum\\), mythology"],
    },
    "fenrir_(smite)": {
        "character": ["fenrir_(smite)"],
        "trigger": ["fenrir \\(smite\\), smite"],
    },
    "kazlee": {"character": ["kazlee"], "trigger": ["kazlee, my little pony"]},
    "june_greenfield": {
        "character": ["june_greenfield"],
        "trigger": ["june greenfield, pokemon"],
    },
    "maya_del_phox": {
        "character": ["maya_del_phox"],
        "trigger": ["maya del phox, pokemon"],
    },
    "rcfox": {"character": ["rcfox"], "trigger": ["rcfox, halloween"]},
    "ceres_(radarn)": {
        "character": ["ceres_(radarn)"],
        "trigger": ["ceres \\(radarn\\), pokemon"],
    },
    "penny_(plantpenetrator)": {
        "character": ["penny_(plantpenetrator)"],
        "trigger": ["penny \\(plantpenetrator\\), pokemon"],
    },
    "plague_knight": {
        "character": ["plague_knight"],
        "trigger": ["plague knight, shovel knight"],
    },
    "daji_(full_bokko_heroes)": {
        "character": ["daji_(full_bokko_heroes)"],
        "trigger": ["daji \\(full bokko heroes\\), drecom"],
    },
    "owen_grady": {
        "character": ["owen_grady"],
        "trigger": ["owen grady, universal studios"],
    },
    "seymore": {"character": ["seymore"], "trigger": ["seymore, easter"]},
    "johnsergal_(character)": {
        "character": ["johnsergal_(character)"],
        "trigger": ["johnsergal \\(character\\), pokemon"],
    },
    "lass_(matsu-sensei)": {
        "character": ["lass_(matsu-sensei)"],
        "trigger": ["lass \\(matsu-sensei\\), golden week"],
    },
    "bailey_(fluff-kevlar)": {
        "character": ["bailey_(fluff-kevlar)"],
        "trigger": ["bailey \\(fluff-kevlar\\), grand theft auto"],
    },
    "honoka_(doa)": {
        "character": ["honoka_(doa)"],
        "trigger": ["honoka \\(doa\\), dead or alive \\(series\\)"],
    },
    "aaron_(undertale)": {
        "character": ["aaron_(undertale)"],
        "trigger": ["aaron \\(undertale\\), undertale \\(series\\)"],
    },
    "luxray_(tdub2217)": {
        "character": ["luxray_(tdub2217)"],
        "trigger": ["luxray \\(tdub2217\\), pokemon"],
    },
    "oberon_(warframe)": {
        "character": ["oberon_(warframe)"],
        "trigger": ["oberon \\(warframe\\), warframe"],
    },
    "evening_breeze": {
        "character": ["evening_breeze"],
        "trigger": ["evening breeze, my little pony"],
    },
    "pyruvic": {"character": ["pyruvic"], "trigger": ["pyruvic, mythology"]},
    "celine_louison": {
        "character": ["celine_louison"],
        "trigger": ["celine louison, my little pony"],
    },
    "mien_(pandashorts)": {
        "character": ["mien_(pandashorts)"],
        "trigger": ["mien \\(pandashorts\\), pokemon"],
    },
    "tag_(rimba_racer)": {
        "character": ["tag_(rimba_racer)"],
        "trigger": ["tag \\(rimba racer\\), rimba racer"],
    },
    "inky_rose_(mlp)": {
        "character": ["inky_rose_(mlp)"],
        "trigger": ["inky rose \\(mlp\\), my little pony"],
    },
    "linkxendo": {"character": ["linkxendo"], "trigger": ["linkxendo, mythology"]},
    "patty_(vimhomeless)": {
        "character": ["patty_(vimhomeless)"],
        "trigger": ["patty \\(vimhomeless\\), mythology"],
    },
    "memory_match": {
        "character": ["memory_match"],
        "trigger": ["memory match, my little pony"],
    },
    "mitchell_(felino)": {
        "character": ["mitchell_(felino)"],
        "trigger": ["mitchell \\(felino\\), nintendo"],
    },
    "demonium": {"character": ["demonium"], "trigger": ["demonium, mythology"]},
    "melanie_(hambor12)": {
        "character": ["melanie_(hambor12)"],
        "trigger": ["melanie \\(hambor12\\), pokemon"],
    },
    "wepawet": {"character": ["wepawet"], "trigger": ["wepawet, pokemon"]},
    "medivh_(soundvariations)": {
        "character": ["medivh_(soundvariations)"],
        "trigger": ["medivh \\(soundvariations\\), mythology"],
    },
    "eva_grimheart": {
        "character": ["eva_grimheart"],
        "trigger": ["eva grimheart, mythology"],
    },
    "jade_jacky_kim": {
        "character": ["jade_jacky_kim"],
        "trigger": ["jade jacky kim, mythology"],
    },
    "minikane": {"character": ["minikane"], "trigger": ["minikane, made in abyss"]},
    "alice_lovage": {
        "character": ["alice_lovage"],
        "trigger": ["alice lovage, mythology"],
    },
    "sophie_(zigzagmag)": {
        "character": ["sophie_(zigzagmag)"],
        "trigger": ["sophie \\(zigzagmag\\), pokemon"],
    },
    "free_(beastars)": {
        "character": ["free_(beastars)"],
        "trigger": ["free \\(beastars\\), beastars"],
    },
    "caylin": {"character": ["caylin"], "trigger": ["caylin, pocky"]},
    "alizea_(blackie94)": {
        "character": ["alizea_(blackie94)"],
        "trigger": ["alizea \\(blackie94\\), mythology"],
    },
    "konomi_(oposa)": {
        "character": ["konomi_(oposa)"],
        "trigger": ["konomi \\(oposa\\), mcdonald's"],
    },
    "ulya_(wjyw)": {
        "character": ["ulya_(wjyw)"],
        "trigger": ["ulya \\(wjyw\\), soyuzmultfilm"],
    },
    "meep_the_kobold_(character)": {
        "character": ["meep_the_kobold_(character)"],
        "trigger": ["meep the kobold \\(character\\), blender cycles"],
    },
    "drake_(kitty_pride)": {
        "character": ["drake_(kitty_pride)"],
        "trigger": ["drake \\(kitty pride\\), kitty pride"],
    },
    "ruby_(ghostth39)": {
        "character": ["ruby_(ghostth39)"],
        "trigger": ["ruby \\(ghostth39\\), pokemon"],
    },
    "kompakt_(kompakt)": {
        "character": ["kompakt_(kompakt)"],
        "trigger": ["kompakt \\(kompakt\\), blender \\(software\\)"],
    },
    "edge_(mario_plus_rabbids)": {
        "character": ["edge_(mario_plus_rabbids)"],
        "trigger": ["edge \\(mario plus rabbids\\), mario plus rabbids sparks of hope"],
    },
    "britz_strudel": {
        "character": ["britz_strudel"],
        "trigger": ["britz strudel, fuga: melodies of steel"],
    },
    "queen_haven_(mlp)": {
        "character": ["queen_haven_(mlp)"],
        "trigger": ["queen haven \\(mlp\\), my little pony"],
    },
    "jellymon_(ghost_game)": {
        "character": ["jellymon_(ghost_game)"],
        "trigger": ["jellymon \\(ghost game\\), digimon"],
    },
    "sibi_(sibi_the_messtress)": {
        "character": ["sibi_(sibi_the_messtress)"],
        "trigger": ["sibi \\(sibi the messtress\\), kinktober"],
    },
    "nikki_(nikki_forever)": {
        "character": ["nikki_(nikki_forever)"],
        "trigger": ["nikki \\(nikki forever\\), mythology"],
    },
    "ariel_(deerkid)": {
        "character": ["ariel_(deerkid)"],
        "trigger": ["ariel \\(deerkid\\), christmas"],
    },
    "alex_(horemheb)": {
        "character": ["alex_(horemheb)"],
        "trigger": ["alex \\(horemheb\\), inkbunny"],
    },
    "hoppy_hopscotch": {
        "character": ["hoppy_hopscotch"],
        "trigger": ["hoppy hopscotch, poppy playtime"],
    },
    "brittany_diggers": {
        "character": ["brittany_diggers"],
        "trigger": ["brittany diggers, gold digger"],
    },
    "tokami": {"character": ["tokami"], "trigger": ["tokami, mythology"]},
    "donner": {"character": ["donner"], "trigger": ["donner, christmas"]},
    "fyxe": {"character": ["fyxe"], "trigger": ["fyxe, nintendo"]},
    "gosalyn_mallard": {
        "character": ["gosalyn_mallard"],
        "trigger": ["gosalyn mallard, disney"],
    },
    "turret_(portal)": {
        "character": ["turret_(portal)"],
        "trigger": ["turret \\(portal\\), valve"],
    },
    "mr._game_and_watch": {
        "character": ["mr._game_and_watch"],
        "trigger": ["mr. game and watch, game and watch"],
    },
    "zorro_re_(character)": {
        "character": ["zorro_re_(character)"],
        "trigger": ["zorro re \\(character\\), christmas"],
    },
    "levor": {"character": ["levor"], "trigger": ["levor, mythology"]},
    "recca": {"character": ["recca"], "trigger": ["recca, mythology"]},
    "lumpy_space_princess": {
        "character": ["lumpy_space_princess"],
        "trigger": ["lumpy space princess, cartoon network"],
    },
    "annie_(lol)": {
        "character": ["annie_(lol)"],
        "trigger": ["annie \\(lol\\), riot games"],
    },
    "bloodhawk_(character)": {
        "character": ["bloodhawk_(character)"],
        "trigger": ["bloodhawk \\(character\\), mythology"],
    },
    "ovni": {"character": ["ovni"], "trigger": ["ovni, disney"]},
    "cait_sith_(ff7)": {
        "character": ["cait_sith_(ff7)"],
        "trigger": ["cait sith \\(ff7\\), square enix"],
    },
    "kinar_(amocin)": {
        "character": ["kinar_(amocin)"],
        "trigger": ["kinar \\(amocin\\), warcraft"],
    },
    "launchpad_mcquack": {
        "character": ["launchpad_mcquack"],
        "trigger": ["launchpad mcquack, ducktales"],
    },
    "bree_(animal_crossing)": {
        "character": ["bree_(animal_crossing)"],
        "trigger": ["bree \\(animal crossing\\), animal crossing"],
    },
    "sam_kensington": {
        "character": ["sam_kensington"],
        "trigger": ["sam kensington, nintendo"],
    },
    "melissa_sweet": {
        "character": ["melissa_sweet"],
        "trigger": ["melissa sweet, mythology"],
    },
    "qwazzy": {"character": ["qwazzy"], "trigger": ["qwazzy, pokemon"]},
    "tsuki_akurei": {
        "character": ["tsuki_akurei"],
        "trigger": ["tsuki akurei, disney"],
    },
    "bangs_(elpatrixf)": {
        "character": ["bangs_(elpatrixf)"],
        "trigger": ["bangs \\(elpatrixf\\), pokemon"],
    },
    "sharpfury_(character)": {
        "character": ["sharpfury_(character)"],
        "trigger": ["sharpfury \\(character\\), gamegod210"],
    },
    "saki_(garasaki)": {
        "character": ["saki_(garasaki)"],
        "trigger": ["saki \\(garasaki\\), mythology"],
    },
    "taree": {"character": ["taree"], "trigger": ["taree, mythology"]},
    "talwyn_apogee": {
        "character": ["talwyn_apogee"],
        "trigger": ["talwyn apogee, sony corporation"],
    },
    "bobbie_(rotten_robbie)": {
        "character": ["bobbie_(rotten_robbie)"],
        "trigger": ["bobbie \\(rotten robbie\\), truly \\(drink\\)"],
    },
    "remidragon": {"character": ["remidragon"], "trigger": ["remidragon, mythology"]},
    "daisy_(doom)": {
        "character": ["daisy_(doom)"],
        "trigger": ["daisy \\(doom\\), id software"],
    },
    "nahyon_(character)": {
        "character": ["nahyon_(character)"],
        "trigger": ["nahyon \\(character\\), mythology"],
    },
    "aquei_(fvt)": {
        "character": ["aquei_(fvt)"],
        "trigger": ["aquei \\(fvt\\), fairies vs tentacles"],
    },
    "nicole_(savestate)": {
        "character": ["nicole_(savestate)"],
        "trigger": ["nicole \\(savestate\\), savestate"],
    },
    "euca_(repeat)": {
        "character": ["euca_(repeat)"],
        "trigger": ["euca \\(repeat\\), repeat \\(visual novel\\)"],
    },
    "tiifu_(the_lion_guard)": {
        "character": ["tiifu_(the_lion_guard)"],
        "trigger": ["tiifu \\(the lion guard\\), disney"],
    },
    "roza_(woadedfox)": {
        "character": ["roza_(woadedfox)"],
        "trigger": ["roza \\(woadedfox\\), christmas"],
    },
    "aaliyah_(oc)": {
        "character": ["aaliyah_(oc)"],
        "trigger": ["aaliyah \\(oc\\), lenny face"],
    },
    "pilgor_(goat_simulator)": {
        "character": ["pilgor_(goat_simulator)"],
        "trigger": ["pilgor \\(goat simulator\\), coffee stain studios"],
    },
    "trix_avenda": {
        "character": ["trix_avenda"],
        "trigger": ["trix avenda, christmas"],
    },
    "otto_the_ott": {
        "character": ["otto_the_ott"],
        "trigger": ["otto the ott, nintendo"],
    },
    "chloe_(iamaneagle)": {
        "character": ["chloe_(iamaneagle)"],
        "trigger": ["chloe \\(iamaneagle\\), the jungle book"],
    },
    "malan_(athiesh)": {
        "character": ["malan_(athiesh)"],
        "trigger": ["malan \\(athiesh\\), meme clothing"],
    },
    "duster_(duster)": {
        "character": ["duster_(duster)"],
        "trigger": ["duster \\(duster\\), star wars"],
    },
    "ezo_red_fox_(kemono_friends)": {
        "character": ["ezo_red_fox_(kemono_friends)"],
        "trigger": ["ezo red fox \\(kemono friends\\), kemono friends"],
    },
    "stiban_(character)": {
        "character": ["stiban_(character)"],
        "trigger": ["stiban \\(character\\), nintendo"],
    },
    "nina_(eigaka)": {
        "character": ["nina_(eigaka)"],
        "trigger": ["nina \\(eigaka\\), mythology"],
    },
    "cid_(fuze)": {"character": ["cid_(fuze)"], "trigger": ["cid \\(fuze\\), pokemon"]},
    "gabriel_(fuze)": {
        "character": ["gabriel_(fuze)"],
        "trigger": ["gabriel \\(fuze\\), hanes"],
    },
    "snow_fawn_poppy_(lol)": {
        "character": ["snow_fawn_poppy_(lol)"],
        "trigger": ["snow fawn poppy \\(lol\\), riot games"],
    },
    "enorach_(0laffson)": {
        "character": ["enorach_(0laffson)"],
        "trigger": ["enorach \\(0laffson\\), christmas"],
    },
    "vitali_advenil": {
        "character": ["vitali_advenil"],
        "trigger": ["vitali advenil, nintendo"],
    },
    "opekun": {"character": ["opekun"], "trigger": ["opekun, pokemon"]},
    "tawny_otterson": {
        "character": ["tawny_otterson"],
        "trigger": ["tawny otterson, disney"],
    },
    "rayne_blanc": {"character": ["rayne_blanc"], "trigger": ["rayne blanc, nintendo"]},
    "cinna_(megacoolbear)": {
        "character": ["cinna_(megacoolbear)"],
        "trigger": ["cinna \\(megacoolbear\\), cartoon network"],
    },
    "fang_(primal)": {
        "character": ["fang_(primal)"],
        "trigger": ["fang \\(primal\\), primal \\(series\\)"],
    },
    "padraig_(masterofall)": {
        "character": ["padraig_(masterofall)"],
        "trigger": ["padraig \\(masterofall\\), disney"],
    },
    "shinamin_ironclaw": {
        "character": ["shinamin_ironclaw"],
        "trigger": ["shinamin ironclaw, warcraft"],
    },
    "corops_blackscale": {
        "character": ["corops_blackscale"],
        "trigger": ["corops blackscale, mythology"],
    },
    "alchiba": {"character": ["alchiba"], "trigger": ["alchiba, lifewonders"]},
    "nora_(zummeng)": {
        "character": ["nora_(zummeng)"],
        "trigger": ["nora \\(zummeng\\), mythology"],
    },
    "soledad_(atomic417)": {
        "character": ["soledad_(atomic417)"],
        "trigger": ["soledad \\(atomic417\\), pokemon"],
    },
    "paul_pfitzner": {
        "character": ["paul_pfitzner"],
        "trigger": ["paul pfitzner, knights college"],
    },
    "timothy_(zer0rebel4)": {
        "character": ["timothy_(zer0rebel4)"],
        "trigger": ["timothy \\(zer0rebel4\\), mythology"],
    },
    "archermagnum": {
        "character": ["archermagnum"],
        "trigger": ["archermagnum, thunderrangers"],
    },
    "redditor_gardevoir": {
        "character": ["redditor_gardevoir"],
        "trigger": ["redditor gardevoir, pokemon"],
    },
    "yetzer_hara": {
        "character": ["yetzer_hara"],
        "trigger": ["yetzer hara, mythology"],
    },
    "flam_kish": {
        "character": ["flam_kish"],
        "trigger": ["flam kish, fuga: melodies of steel"],
    },
    "liam_(codymathews)": {
        "character": ["liam_(codymathews)"],
        "trigger": ["liam \\(codymathews\\), catchingup"],
    },
    "ming_lee_(turning_red)": {
        "character": ["ming_lee_(turning_red)"],
        "trigger": ["ming lee \\(turning red\\), disney"],
    },
    "schorl_(kitfox-crimson)": {
        "character": ["schorl_(kitfox-crimson)"],
        "trigger": ["schorl \\(kitfox-crimson\\), in our shadow"],
    },
    "elizabeth_mendoza": {
        "character": ["elizabeth_mendoza"],
        "trigger": ["elizabeth mendoza, among us"],
    },
    "cassie_(fnaf)": {
        "character": ["cassie_(fnaf)"],
        "trigger": ["cassie \\(fnaf\\), five nights at freddy's: security breach"],
    },
    "fern_(frieren)": {
        "character": ["fern_(frieren)"],
        "trigger": ["fern \\(frieren\\), frieren beyond journey's end"],
    },
    "pal_tamer": {"character": ["pal_tamer"], "trigger": ["pal tamer, palworld"]},
    "kappy_(character)": {
        "character": ["kappy_(character)"],
        "trigger": ["kappy \\(character\\), mythology"],
    },
    "lucifer": {"character": ["lucifer"], "trigger": ["lucifer, convent of hell"]},
    "razr_(character)": {
        "character": ["razr_(character)"],
        "trigger": ["razr \\(character\\), mythology"],
    },
    "lacey_(meesh)": {
        "character": ["lacey_(meesh)"],
        "trigger": ["lacey \\(meesh\\), the valet and the vixen"],
    },
    "colleen_(sugarnutz)": {
        "character": ["colleen_(sugarnutz)"],
        "trigger": ["colleen \\(sugarnutz\\), mythology"],
    },
    "calamity_coyote": {
        "character": ["calamity_coyote"],
        "trigger": ["calamity coyote, warner brothers"],
    },
    "the_white_rabbit": {
        "character": ["the_white_rabbit"],
        "trigger": ["the white rabbit, alice in wonderland"],
    },
    "mr._mephit": {"character": ["mr._mephit"], "trigger": ["mr. mephit, mythology"]},
    "linkin": {"character": ["linkin"], "trigger": ["linkin, mythology"]},
    "frisky-lime": {"character": ["frisky-lime"], "trigger": ["frisky-lime, nintendo"]},
    "chance_(bad_dragon)": {
        "character": ["chance_(bad_dragon)"],
        "trigger": ["chance \\(bad dragon\\), bad dragon"],
    },
    "aurora_spencer": {
        "character": ["aurora_spencer"],
        "trigger": ["aurora spencer, christmas"],
    },
    "justin-parallax": {
        "character": ["justin-parallax"],
        "trigger": ["justin-parallax, mythology"],
    },
    "ammy": {"character": ["ammy"], "trigger": ["ammy, mythology"]},
    "sue_ellen_armstrong": {
        "character": ["sue_ellen_armstrong"],
        "trigger": ["sue ellen armstrong, arthur \\(series\\)"],
    },
    "death_knight_(warcraft)": {
        "character": ["death_knight_(warcraft)"],
        "trigger": ["death knight \\(warcraft\\), warcraft"],
    },
    "lizanne": {
        "character": ["lizanne"],
        "trigger": ["lizanne, blender \\(software\\)"],
    },
    "reva": {"character": ["reva"], "trigger": ["reva, warcraft"]},
    "soundwave": {"character": ["soundwave"], "trigger": ["soundwave, takara tomy"]},
    "buck_(evane)": {
        "character": ["buck_(evane)"],
        "trigger": ["buck \\(evane\\), evane"],
    },
    "wan_wu": {"character": ["wan_wu"], "trigger": ["wan wu, kung fu panda"]},
    "opal_(ashnar)": {
        "character": ["opal_(ashnar)"],
        "trigger": ["opal \\(ashnar\\), mythology"],
    },
    "sawyer_(ferobird)": {
        "character": ["sawyer_(ferobird)"],
        "trigger": ["sawyer \\(ferobird\\), mythology"],
    },
    "hazel": {"character": ["hazel"], "trigger": ["hazel, mythology"]},
    "zapp_(mlp)": {
        "character": ["zapp_(mlp)"],
        "trigger": ["zapp \\(mlp\\), my little pony"],
    },
    "bose_(character)": {
        "character": ["bose_(character)"],
        "trigger": ["bose \\(character\\), mythology"],
    },
    "tahla_(tahla)": {
        "character": ["tahla_(tahla)"],
        "trigger": ["tahla \\(tahla\\), pokemon"],
    },
    "storm_(stormblazer)": {
        "character": ["storm_(stormblazer)"],
        "trigger": ["storm \\(stormblazer\\), mythology"],
    },
    "purrcules_(character)": {
        "character": ["purrcules_(character)"],
        "trigger": ["purrcules \\(character\\), apple inc."],
    },
    "spring_beauty": {
        "character": ["spring_beauty"],
        "trigger": ["spring beauty, mythology"],
    },
    "daria_lang": {"character": ["daria_lang"], "trigger": ["daria lang, pokemon"]},
    "prea": {"character": ["prea"], "trigger": ["prea, christmas"]},
    "ryan_carthage": {
        "character": ["ryan_carthage"],
        "trigger": ["ryan carthage, mythology"],
    },
    "beau_(williamca)": {
        "character": ["beau_(williamca)"],
        "trigger": ["beau \\(williamca\\), pokemon"],
    },
    "beardo_(animal_crossing)": {
        "character": ["beardo_(animal_crossing)"],
        "trigger": ["beardo \\(animal crossing\\), animal crossing"],
    },
    "loganhen": {"character": ["loganhen"], "trigger": ["loganhen, mass effect"]},
    "ratatoskr": {"character": ["ratatoskr"], "trigger": ["ratatoskr, smite"]},
    "saerro": {"character": ["saerro"], "trigger": ["saerro, mythology"]},
    "kenzie": {"character": ["kenzie"], "trigger": ["kenzie, denver broncos"]},
    "ray-bleiz": {"character": ["ray-bleiz"], "trigger": ["ray-bleiz, mythology"]},
    "max_midnight": {
        "character": ["max_midnight"],
        "trigger": ["max midnight, disney"],
    },
    "trinity_(shenzel)": {
        "character": ["trinity_(shenzel)"],
        "trigger": ["trinity \\(shenzel\\), mythology"],
    },
    "courtney_brushmarke": {
        "character": ["courtney_brushmarke"],
        "trigger": ["courtney brushmarke, disney"],
    },
    "sheazu": {"character": ["sheazu"], "trigger": ["sheazu, mythology"]},
    "areumi_(zinfyu)": {
        "character": ["areumi_(zinfyu)"],
        "trigger": ["areumi \\(zinfyu\\), pokemon"],
    },
    "zabaniyya_(tas)": {
        "character": ["zabaniyya_(tas)"],
        "trigger": ["zabaniyya \\(tas\\), lifewonders"],
    },
    "bacon_(baconbakin)": {
        "character": ["bacon_(baconbakin)"],
        "trigger": ["bacon \\(baconbakin\\), mythology"],
    },
    "d-class": {"character": ["d-class"], "trigger": ["d-class, scp foundation"]},
    "grimfaust_(nightterror)": {
        "character": ["grimfaust_(nightterror)"],
        "trigger": ["grimfaust \\(nightterror\\), mythology"],
    },
    "dj_strap": {"character": ["dj_strap"], "trigger": ["dj strap, mythology"]},
    "mark_beaks": {"character": ["mark_beaks"], "trigger": ["mark beaks, disney"]},
    "amanda_(smile4amanda)": {
        "character": ["amanda_(smile4amanda)"],
        "trigger": ["amanda \\(smile4amanda\\), playful distractions"],
    },
    "katty_hupokoro": {
        "character": ["katty_hupokoro"],
        "trigger": ["katty hupokoro, halloween"],
    },
    "rosaline_(bronx23)": {
        "character": ["rosaline_(bronx23)"],
        "trigger": ["rosaline \\(bronx23\\), bronx23"],
    },
    "starbuck": {"character": ["starbuck"], "trigger": ["starbuck, mythology"]},
    "joshy_(kibaru)": {
        "character": ["joshy_(kibaru)"],
        "trigger": ["joshy \\(kibaru\\), pokemon"],
    },
    "ronnie_(yinller)": {
        "character": ["ronnie_(yinller)"],
        "trigger": ["ronnie \\(yinller\\), angel in the forest"],
    },
    "magntasona": {"character": ["magntasona"], "trigger": ["magntasona, nintendo"]},
    "dallas_(101_dalmatians)": {
        "character": ["dallas_(101_dalmatians)"],
        "trigger": ["dallas \\(101 dalmatians\\), disney"],
    },
    "maisie_whisk": {
        "character": ["maisie_whisk"],
        "trigger": ["maisie whisk, valentine's day"],
    },
    "rayne_(quin-nsfw)": {
        "character": ["rayne_(quin-nsfw)"],
        "trigger": ["rayne \\(quin-nsfw\\), greek mythology"],
    },
    "cassie_cooper": {"character": ["cassie_cooper"], "trigger": ["cassie cooper"]},
    "purrcival": {
        "character": ["purrcival"],
        "trigger": ["purrcival, cartoon network"],
    },
    "helga_(world_flipper)": {
        "character": ["helga_(world_flipper)"],
        "trigger": ["helga \\(world flipper\\), cygames"],
    },
    "queen_vinyl_da.i'gyu-kazotetsu": {
        "character": ["queen_vinyl_da.i'gyu-kazotetsu"],
        "trigger": ["queen vinyl da.i'gyu-kazotetsu, mythology"],
    },
    "valexia": {"character": ["valexia"], "trigger": ["valexia, mythology"]},
    "nina_flip": {"character": ["nina_flip"], "trigger": ["nina flip, studio trigger"]},
    "fay_(yvem)": {
        "character": ["fay_(yvem)"],
        "trigger": ["fay \\(yvem\\), christmas"],
    },
    "totally_tubular_coco": {
        "character": ["totally_tubular_coco"],
        "trigger": ["totally tubular coco, crash bandicoot \\(series\\)"],
    },
    "lotte_(cobalt_snow)": {
        "character": ["lotte_(cobalt_snow)"],
        "trigger": ["lotte \\(cobalt snow\\), okay \\(meme\\)"],
    },
    "wen_kamui_(tas)": {
        "character": ["wen_kamui_(tas)"],
        "trigger": ["wen kamui \\(tas\\), lifewonders"],
    },
    "li_bing_(white_cat_legend)": {
        "character": ["li_bing_(white_cat_legend)"],
        "trigger": ["li bing \\(white cat legend\\), white cat legend"],
    },
    "username_(character)": {
        "character": ["username_(character)"],
        "trigger": ["username \\(character\\), blender \\(software\\)"],
    },
    "abella_mf_spirit": {
        "character": ["abella_mf_spirit"],
        "trigger": ["abella mf spirit, xbox game studios"],
    },
    "gracie_(vixeyhuskybutt)": {
        "character": ["gracie_(vixeyhuskybutt)"],
        "trigger": ["gracie \\(vixeyhuskybutt\\), east asian mythology"],
    },
    "long_the_dragon": {
        "character": ["long_the_dragon"],
        "trigger": ["long the dragon, deepest sword"],
    },
    "mitsy_(itsymitsy)": {
        "character": ["mitsy_(itsymitsy)"],
        "trigger": ["mitsy \\(itsymitsy\\), mythology"],
    },
    "fuckable_pin": {
        "character": ["fuckable_pin"],
        "trigger": ["fuckable pin, wyer bowling \\(meme\\)"],
    },
    "thunk_(gyro)": {
        "character": ["thunk_(gyro)"],
        "trigger": ["thunk \\(gyro\\), pokemon"],
    },
    "ajizza_(ajizza)": {
        "character": ["ajizza_(ajizza)"],
        "trigger": ["ajizza \\(ajizza\\), spiral knights"],
    },
    "raelynn_(mynka)": {
        "character": ["raelynn_(mynka)"],
        "trigger": ["raelynn \\(mynka\\)"],
    },
    "gen_kiryu": {
        "character": ["gen_kiryu"],
        "trigger": ["gen kiryu, menacing \\(meme\\)"],
    },
    "paul_grayson": {
        "character": ["paul_grayson"],
        "trigger": ["paul grayson, pokemon"],
    },
    "ra'deer": {"character": ["ra'deer"], "trigger": ["ra'deer, mythology"]},
    "jasmin_(jasminthemanticore)": {
        "character": ["jasmin_(jasminthemanticore)"],
        "trigger": ["jasmin \\(jasminthemanticore\\), star fox"],
    },
    "gami_cross": {"character": ["gami_cross"], "trigger": ["gami cross, nintendo"]},
    "preyfar": {"character": ["preyfar"], "trigger": ["preyfar, mythology"]},
    "bucky_o'hare": {
        "character": ["bucky_o'hare"],
        "trigger": ["bucky o'hare, bucky o'hare \\(series\\)"],
    },
    "summer_grass": {
        "character": ["summer_grass"],
        "trigger": ["summer grass, mythology"],
    },
    "chester_cheetah": {
        "character": ["chester_cheetah"],
        "trigger": ["chester cheetah, cheetos"],
    },
    "durga_(tas)": {
        "character": ["durga_(tas)"],
        "trigger": ["durga \\(tas\\), lifewonders"],
    },
    "sir_hiss": {"character": ["sir_hiss"], "trigger": ["sir hiss, disney"]},
    "davin": {"character": ["davin"], "trigger": ["davin, las lindas"]},
    "immelmann_(character)": {
        "character": ["immelmann_(character)"],
        "trigger": ["immelmann \\(character\\), mythology"],
    },
    "bill_grey": {"character": ["bill_grey"], "trigger": ["bill grey, star fox"]},
    "roodaka": {"character": ["roodaka"], "trigger": ["roodaka, bionicle"]},
    "cleo_(yutrah)": {
        "character": ["cleo_(yutrah)"],
        "trigger": ["cleo \\(yutrah\\), texas longhorns"],
    },
    "dasher": {"character": ["dasher"], "trigger": ["dasher, christmas"]},
    "turmoil_(swat_kats)": {
        "character": ["turmoil_(swat_kats)"],
        "trigger": ["turmoil \\(swat kats\\), swat kats"],
    },
    "kate_(morpheuskibbe)": {
        "character": ["kate_(morpheuskibbe)"],
        "trigger": ["kate \\(morpheuskibbe\\), morpheuskibbe"],
    },
    "kayle_(lol)": {
        "character": ["kayle_(lol)"],
        "trigger": ["kayle \\(lol\\), riot games"],
    },
    "xaenyth_(character)": {
        "character": ["xaenyth_(character)"],
        "trigger": ["xaenyth \\(character\\), undertale \\(series\\)"],
    },
    "miles_yellow": {
        "character": ["miles_yellow"],
        "trigger": ["miles yellow, sonic the hedgehog \\(series\\)"],
    },
    "ipsywitch": {"character": ["ipsywitch"], "trigger": ["ipsywitch, my little pony"]},
    "evalion_(character)": {
        "character": ["evalion_(character)"],
        "trigger": ["evalion \\(character\\), mythology"],
    },
    "clef": {"character": ["clef"], "trigger": ["clef, mythology"]},
    "katinka": {"character": ["katinka"], "trigger": ["katinka, mythology"]},
    "valheru": {"character": ["valheru"], "trigger": ["valheru, halloween"]},
    "kerub_crepin": {
        "character": ["kerub_crepin"],
        "trigger": ["kerub crepin, ankama"],
    },
    "mesa_(warframe)": {
        "character": ["mesa_(warframe)"],
        "trigger": ["mesa \\(warframe\\), warframe"],
    },
    "vanilla_(canary)": {
        "character": ["vanilla_(canary)"],
        "trigger": ["vanilla \\(canary\\), christmas"],
    },
    "duke_corgi": {"character": ["duke_corgi"], "trigger": ["duke corgi, heineken"]},
    "kyle_r._fish": {
        "character": ["kyle_r._fish"],
        "trigger": ["kyle r. fish, mythology"],
    },
    "slave_pup_(marimo)": {
        "character": ["slave_pup_(marimo)"],
        "trigger": ["slave pup \\(marimo\\), alien \\(franchise\\)"],
    },
    "phos_(phosaggro)": {
        "character": ["phos_(phosaggro)"],
        "trigger": ["phos \\(phosaggro\\), pokemon"],
    },
    "rose_(mewmus)": {
        "character": ["rose_(mewmus)"],
        "trigger": ["rose \\(mewmus\\), nintendo"],
    },
    "ashtoreth_illacrimo": {
        "character": ["ashtoreth_illacrimo"],
        "trigger": ["ashtoreth illacrimo, mythology"],
    },
    "shrimp_(uk_brony)": {
        "character": ["shrimp_(uk_brony)"],
        "trigger": ["shrimp \\(uk brony\\), male swimwear challenge"],
    },
    "vivian_vivi": {
        "character": ["vivian_vivi"],
        "trigger": ["vivian vivi, april fools' day"],
    },
    "nash_(zenirix)": {
        "character": ["nash_(zenirix)"],
        "trigger": ["nash \\(zenirix\\), mythology"],
    },
    "mike_(sigma_x)": {
        "character": ["mike_(sigma_x)"],
        "trigger": ["mike \\(sigma x\\), patreon"],
    },
    "felipunny": {"character": ["felipunny"], "trigger": ["felipunny, pokemon"]},
    "great_kitsune_(housepets!)": {
        "character": ["great_kitsune_(housepets!)"],
        "trigger": ["great kitsune \\(housepets!\\), housepets!"],
    },
    "teo_(hayakain)": {
        "character": ["teo_(hayakain)"],
        "trigger": ["teo \\(hayakain\\), sonic the hedgehog \\(series\\)"],
    },
    "garcia": {"character": ["garcia"], "trigger": ["garcia, nintendo"]},
    "shadi_(0laffson)": {
        "character": ["shadi_(0laffson)"],
        "trigger": ["shadi \\(0laffson\\), smug cat"],
    },
    "tiffany_brewwer": {
        "character": ["tiffany_brewwer"],
        "trigger": ["tiffany brewwer, if hell had a taste"],
    },
    "fennec_fox_(kemono_friends)": {
        "character": ["fennec_fox_(kemono_friends)"],
        "trigger": ["fennec fox \\(kemono friends\\), kemono friends"],
    },
    "nicholas_c._corbin": {
        "character": ["nicholas_c._corbin"],
        "trigger": ["nicholas c. corbin, the corbin family"],
    },
    "koji_koda": {
        "character": ["koji_koda"],
        "trigger": ["koji koda, my hero academia"],
    },
    "novah_ikaro_(character)": {
        "character": ["novah_ikaro_(character)"],
        "trigger": ["novah ikaro \\(character\\), mythology"],
    },
    "sate": {"character": ["sate"], "trigger": ["sate, mythology"]},
    "matthewwoodward": {
        "character": ["matthewwoodward"],
        "trigger": ["matthewwoodward, blinkblinkblink"],
    },
    "kivu": {"character": ["kivu"], "trigger": ["kivu, gopro"]},
    "dog_girl_(berseepon09)": {
        "character": ["dog_girl_(berseepon09)"],
        "trigger": ["dog girl \\(berseepon09\\), mythology"],
    },
    "ubzerd_(character)": {
        "character": ["ubzerd_(character)"],
        "trigger": ["ubzerd \\(character\\), blender \\(software\\)"],
    },
    "rani_(the_lion_guard)": {
        "character": ["rani_(the_lion_guard)"],
        "trigger": ["rani \\(the lion guard\\), disney"],
    },
    "aerrow": {"character": ["aerrow"], "trigger": ["aerrow, mythology"]},
    "retter": {"character": ["retter"], "trigger": ["retter, mythology"]},
    "dr_rabbit_(tomtc)": {
        "character": ["dr_rabbit_(tomtc)"],
        "trigger": ["dr rabbit \\(tomtc\\), mythology"],
    },
    "lilika_snowheart": {
        "character": ["lilika_snowheart"],
        "trigger": ["lilika snowheart, nintendo"],
    },
    "tinsel_(wanderlust)": {
        "character": ["tinsel_(wanderlust)"],
        "trigger": ["tinsel \\(wanderlust\\), pokemon"],
    },
    "lira_(remix1997)": {
        "character": ["lira_(remix1997)"],
        "trigger": ["lira \\(remix1997\\), mythology"],
    },
    "karou_(thekbear)": {
        "character": ["karou_(thekbear)"],
        "trigger": ["karou \\(thekbear\\), nintendo"],
    },
    "zhima_(diives)": {
        "character": ["zhima_(diives)"],
        "trigger": ["zhima \\(diives\\), xingzuo temple"],
    },
    "viv_(lowkeytoast)": {
        "character": ["viv_(lowkeytoast)"],
        "trigger": ["viv \\(lowkeytoast\\), pokemon"],
    },
    "nuggy_(anaugi)": {
        "character": ["nuggy_(anaugi)"],
        "trigger": ["nuggy \\(anaugi\\), nintendo"],
    },
    "jason_(mr5star)": {
        "character": ["jason_(mr5star)"],
        "trigger": ["jason \\(mr5star\\), halloween"],
    },
    "surisu_(tohupo)": {
        "character": ["surisu_(tohupo)"],
        "trigger": ["surisu \\(tohupo\\), halloween"],
    },
    "majiro_the_hedgehog": {
        "character": ["majiro_the_hedgehog"],
        "trigger": ["majiro the hedgehog, sleepy princess in the demon castle"],
    },
    "rubi_(stemingbunbun)": {
        "character": ["rubi_(stemingbunbun)"],
        "trigger": ["rubi \\(stemingbunbun\\), pokemon"],
    },
    "hee-na": {"character": ["hee-na"], "trigger": ["hee-na, mythology"]},
    "kyo_(stargazer)": {
        "character": ["kyo_(stargazer)"],
        "trigger": ["kyo \\(stargazer\\), to be continued"],
    },
    "silver_fang_(furry1997)": {
        "character": ["silver_fang_(furry1997)"],
        "trigger": ["silver fang \\(furry1997\\), mythology"],
    },
    "osiris_callisto": {
        "character": ["osiris_callisto"],
        "trigger": ["osiris callisto, mythology"],
    },
    "lyra_(w4g4)": {
        "character": ["lyra_(w4g4)"],
        "trigger": ["lyra \\(w4g4\\), mythology"],
    },
    "derpybelle": {
        "character": ["derpybelle"],
        "trigger": ["derpybelle, animal crossing"],
    },
    "fifi_(somemf)": {
        "character": ["fifi_(somemf)"],
        "trigger": ["fifi \\(somemf\\), christmas"],
    },
    "cassandra_(ozoneserpent)": {
        "character": ["cassandra_(ozoneserpent)"],
        "trigger": ["cassandra \\(ozoneserpent\\), star fox"],
    },
    "sam_(zootopia+)": {
        "character": ["sam_(zootopia+)"],
        "trigger": ["sam \\(zootopia+\\), disney"],
    },
    "roxy_(world_of_ruan)": {
        "character": ["roxy_(world_of_ruan)"],
        "trigger": ["roxy \\(world of ruan\\), world of ruan"],
    },
    "skyshadow_(character)": {
        "character": ["skyshadow_(character)"],
        "trigger": ["skyshadow \\(character\\), mythology"],
    },
    "maxine_blackbunny": {
        "character": ["maxine_blackbunny"],
        "trigger": ["maxine blackbunny, canada day"],
    },
    "blinx": {"character": ["blinx"], "trigger": ["blinx, microsoft"]},
    "popka": {"character": ["popka"], "trigger": ["popka, bandai namco"]},
    "satsuki_kiryuin": {
        "character": ["satsuki_kiryuin"],
        "trigger": ["satsuki kiryuin, kill la kill"],
    },
    "konata_izumi": {
        "character": ["konata_izumi"],
        "trigger": ["konata izumi, lucky star"],
    },
    "chibiterasu_(okami)": {
        "character": ["chibiterasu_(okami)"],
        "trigger": ["chibiterasu \\(okami\\), okami \\(capcom\\)"],
    },
    "krusha": {
        "character": ["krusha"],
        "trigger": ["krusha, donkey kong \\(series\\)"],
    },
    "koopie_koo": {"character": ["koopie_koo"], "trigger": ["koopie koo, mario bros"]},
    "poke_kid": {"character": ["poke_kid"], "trigger": ["poke kid, pokemon"]},
    "rash_(battletoads)": {
        "character": ["rash_(battletoads)"],
        "trigger": ["rash \\(battletoads\\), rareware"],
    },
    "hamton_j._pig": {
        "character": ["hamton_j._pig"],
        "trigger": ["hamton j. pig, warner brothers"],
    },
    "niis_(character)": {
        "character": ["niis_(character)"],
        "trigger": ["niis \\(character\\), valentine's day"],
    },
    "kandace_(sugarnutz)": {
        "character": ["kandace_(sugarnutz)"],
        "trigger": ["kandace \\(sugarnutz\\), flinters' roomies"],
    },
    "nickolai_alaric": {
        "character": ["nickolai_alaric"],
        "trigger": ["nickolai alaric, twokinds"],
    },
    "teryx_(dinosaucers)": {
        "character": ["teryx_(dinosaucers)"],
        "trigger": ["teryx \\(dinosaucers\\), dinosaucers"],
    },
    "ashiji_(character)": {
        "character": ["ashiji_(character)"],
        "trigger": ["ashiji \\(character\\), mythology"],
    },
    "shasta": {"character": ["shasta"], "trigger": ["shasta, digimon"]},
    "siegfried": {"character": ["siegfried"], "trigger": ["siegfried, mythology"]},
    "dairou": {"character": ["dairou"], "trigger": ["dairou, pokemon"]},
    "laini": {"character": ["laini"], "trigger": ["laini, nintendo"]},
    "coop_(wrng)": {
        "character": ["coop_(wrng)"],
        "trigger": ["coop \\(wrng\\), wolf's rain next generation"],
    },
    "natsume_(wrng)": {
        "character": ["natsume_(wrng)"],
        "trigger": ["natsume \\(wrng\\), wolf's rain next generation"],
    },
    "jojo_(coc)": {
        "character": ["jojo_(coc)"],
        "trigger": ["jojo \\(coc\\), corruption of champions"],
    },
    "blackriver": {"character": ["blackriver"], "trigger": ["blackriver, ctenophorae"]},
    "kaijumi": {"character": ["kaijumi"], "trigger": ["kaijumi, pokemon"]},
    "phina_(ashnar)": {
        "character": ["phina_(ashnar)"],
        "trigger": ["phina \\(ashnar\\), mythology"],
    },
    "rocket_grunt": {
        "character": ["rocket_grunt"],
        "trigger": ["rocket grunt, team rocket"],
    },
    "davis_motomiya": {
        "character": ["davis_motomiya"],
        "trigger": ["davis motomiya, digimon"],
    },
    "gustav": {"character": ["gustav"], "trigger": ["gustav, mythology"]},
    "meushi_mattie_(matsu-sensei)": {
        "character": ["meushi_mattie_(matsu-sensei)"],
        "trigger": ["meushi mattie \\(matsu-sensei\\), touch the cow do it now"],
    },
    "dragonslayer_ornstein": {
        "character": ["dragonslayer_ornstein"],
        "trigger": ["dragonslayer ornstein, fromsoftware"],
    },
    "ulric_arnoux": {
        "character": ["ulric_arnoux"],
        "trigger": ["ulric arnoux, fahleir"],
    },
    "wanda_werewolf": {
        "character": ["wanda_werewolf"],
        "trigger": ["wanda werewolf, hotel transylvania"],
    },
    "anselm_fenrisulfr": {
        "character": ["anselm_fenrisulfr"],
        "trigger": ["anselm fenrisulfr, mythology"],
    },
    "fynath": {"character": ["fynath"], "trigger": ["fynath, mythology"]},
    "sketchy_skylar_(character)": {
        "character": ["sketchy_skylar_(character)"],
        "trigger": ["sketchy skylar \\(character\\), my little pony"],
    },
    "marionette_(psychojohn2)": {
        "character": ["marionette_(psychojohn2)"],
        "trigger": ["marionette \\(psychojohn2\\), five nights at freddy's"],
    },
    "tamira_(rimba_racer)": {
        "character": ["tamira_(rimba_racer)"],
        "trigger": ["tamira \\(rimba racer\\), rimba racer"],
    },
    "frederic_(david_siegl)": {
        "character": ["frederic_(david_siegl)"],
        "trigger": ["frederic \\(david siegl\\)"],
    },
    "mettle_winslowe": {
        "character": ["mettle_winslowe"],
        "trigger": ["mettle winslowe, nintendo"],
    },
    "ara_(kin)": {"character": ["ara_(kin)"], "trigger": ["ara \\(kin\\), mythology"]},
    "hali_(character)": {
        "character": ["hali_(character)"],
        "trigger": ["hali \\(character\\), samurai jack"],
    },
    "selmers_ann_forrester": {
        "character": ["selmers_ann_forrester"],
        "trigger": ["selmers ann forrester, night in the woods"],
    },
    "dyspo": {"character": ["dyspo"], "trigger": ["dyspo, dragon ball"]},
    "rosiesquish": {
        "character": ["rosiesquish"],
        "trigger": ["rosiesquish, mythology"],
    },
    "zachary_(lord_salt)": {
        "character": ["zachary_(lord_salt)"],
        "trigger": ["zachary \\(lord salt\\), mythology"],
    },
    "krek_(krekk0v)": {
        "character": ["krek_(krekk0v)"],
        "trigger": ["krek \\(krekk0v\\), haydee \\(game\\)"],
    },
    "joji_(@jojipando)": {
        "character": ["joji_(@jojipando)"],
        "trigger": ["joji \\(@jojipando\\), mythology"],
    },
    "sorto": {"character": ["sorto"], "trigger": ["sorto, christmas"]},
    "krilinus_sixus": {
        "character": ["krilinus_sixus"],
        "trigger": ["krilinus sixus, mythology"],
    },
    "river_lakes": {"character": ["river_lakes"], "trigger": ["river lakes, wikihow"]},
    "terramar_(mlp)": {
        "character": ["terramar_(mlp)"],
        "trigger": ["terramar \\(mlp\\), my little pony"],
    },
    "sammy_(hambor12)": {
        "character": ["sammy_(hambor12)"],
        "trigger": ["sammy \\(hambor12\\), pokemon"],
    },
    "rufuta_hachitani": {
        "character": ["rufuta_hachitani"],
        "trigger": ["rufuta hachitani, working buddies!"],
    },
    "silver_(killerwolf1020)": {
        "character": ["silver_(killerwolf1020)"],
        "trigger": ["silver \\(killerwolf1020\\), mythology"],
    },
    "henri_(r3drunner)": {
        "character": ["henri_(r3drunner)"],
        "trigger": ["henri \\(r3drunner\\), mythology"],
    },
    "flaffy_(viskasunya)": {
        "character": ["flaffy_(viskasunya)"],
        "trigger": ["flaffy \\(viskasunya\\), meme clothing"],
    },
    "c_(chirmaya)": {
        "character": ["c_(chirmaya)"],
        "trigger": ["c \\(chirmaya\\), chirmaya"],
    },
    "yvette_(kilinah)": {
        "character": ["yvette_(kilinah)"],
        "trigger": ["yvette \\(kilinah\\), christmas"],
    },
    "pepper_(kittyprint)": {
        "character": ["pepper_(kittyprint)"],
        "trigger": ["pepper \\(kittyprint\\), nintendo switch"],
    },
    "roy_blake": {"character": ["roy_blake"], "trigger": ["roy blake, pocky"]},
    "dahsharky_(character)": {
        "character": ["dahsharky_(character)"],
        "trigger": ["dahsharky \\(character\\), source filmmaker"],
    },
    "bael_(tas)": {
        "character": ["bael_(tas)"],
        "trigger": ["bael \\(tas\\), lifewonders"],
    },
    "goh_(pokemon)": {
        "character": ["goh_(pokemon)"],
        "trigger": ["goh \\(pokemon\\), pokemon"],
    },
    "berr": {"character": ["berr"], "trigger": ["berr, pokemon"]},
    "billie_corneja": {
        "character": ["billie_corneja"],
        "trigger": ["billie corneja, mythology"],
    },
    "bloot_(bloot)": {
        "character": ["bloot_(bloot)"],
        "trigger": ["bloot \\(bloot\\), mythology"],
    },
    "orion_(xeoniios)": {
        "character": ["orion_(xeoniios)"],
        "trigger": ["orion \\(xeoniios\\), mythology"],
    },
    "felix_(nik159)": {
        "character": ["felix_(nik159)"],
        "trigger": ["felix \\(nik159\\), christmas"],
    },
    "robo_fizzarolli": {
        "character": ["robo_fizzarolli"],
        "trigger": ["robo fizzarolli, helluva boss"],
    },
    "vulpecula": {"character": ["vulpecula"], "trigger": ["vulpecula, lifewonders"]},
    "olannah": {"character": ["olannah"], "trigger": ["olannah"]},
    "reeda": {"character": ["reeda"], "trigger": ["reeda, goodbye volcano high"]},
    "chernobog_kuzarnak": {
        "character": ["chernobog_kuzarnak"],
        "trigger": ["chernobog kuzarnak, mythology"],
    },
    "trinity_(farran_height)": {
        "character": ["trinity_(farran_height)"],
        "trigger": ["trinity \\(farran height\\), pokemon"],
    },
    "hack_montblanc": {
        "character": ["hack_montblanc"],
        "trigger": ["hack montblanc, fuga: melodies of steel"],
    },
    "ash_(tfwnocatgirlgf)": {
        "character": ["ash_(tfwnocatgirlgf)"],
        "trigger": ["ash \\(tfwnocatgirlgf\\), nintendo"],
    },
    "mavis_delcat": {
        "character": ["mavis_delcat"],
        "trigger": ["mavis delcat, mythology"],
    },
    "zacharoff_(anothereidos_r)": {
        "character": ["zacharoff_(anothereidos_r)"],
        "trigger": ["zacharoff \\(anothereidos r\\), another eidos of dragon vein r"],
    },
    "izabell_carroll_(forestdale)": {
        "character": ["izabell_carroll_(forestdale)"],
        "trigger": ["izabell carroll \\(forestdale\\), forestdale"],
    },
    "duncan_(kitfox_crimson)": {
        "character": ["duncan_(kitfox_crimson)"],
        "trigger": ["duncan \\(kitfox crimson\\), stolen generation"],
    },
    "cassandra_(snoot_game)": {
        "character": ["cassandra_(snoot_game)"],
        "trigger": ["cassandra \\(snoot game\\), cavemanon studios"],
    },
    "neve_lunn": {"character": ["neve_lunn"], "trigger": ["neve lunn, mythology"]},
    "nala_(mayosplash)": {
        "character": ["nala_(mayosplash)"],
        "trigger": ["nala \\(mayosplash\\), disney"],
    },
    "mike_blade": {"character": ["mike_blade"], "trigger": ["mike blade, lovely pets"]},
    "giroro": {"character": ["giroro"], "trigger": ["giroro, sgt. frog"]},
    "yosuke_hanamura": {
        "character": ["yosuke_hanamura"],
        "trigger": ["yosuke hanamura, sega"],
    },
    "toph_beifong": {
        "character": ["toph_beifong"],
        "trigger": ["toph beifong, avatar: the last airbender"],
    },
    "drip_(jack)": {
        "character": ["drip_(jack)"],
        "trigger": ["drip \\(jack\\), jack \\(webcomic\\)"],
    },
    "splendid_(htf)": {
        "character": ["splendid_(htf)"],
        "trigger": ["splendid \\(htf\\), happy tree friends"],
    },
    "geronimo_stilton": {
        "character": ["geronimo_stilton"],
        "trigger": ["geronimo stilton, geronimo stilton \\(series\\)"],
    },
    "eve_(wall-e)": {
        "character": ["eve_(wall-e)"],
        "trigger": ["eve \\(wall-e\\), disney"],
    },
    "kyle_(redrusker)": {
        "character": ["kyle_(redrusker)"],
        "trigger": ["kyle \\(redrusker\\), the deep dark"],
    },
    "woody_woodpecker": {
        "character": ["woody_woodpecker"],
        "trigger": ["woody woodpecker, the woody woodpecker show"],
    },
    "bianca_(pokemon)": {
        "character": ["bianca_(pokemon)"],
        "trigger": ["bianca \\(pokemon\\), pokemon"],
    },
    "silver_fang": {
        "character": ["silver_fang"],
        "trigger": ["silver fang, mythology"],
    },
    "lafille": {"character": ["lafille"], "trigger": ["lafille, nintendo"]},
    "tiamat_(dnd)": {
        "character": ["tiamat_(dnd)"],
        "trigger": ["tiamat \\(dnd\\), mythology"],
    },
    "charme": {"character": ["charme"], "trigger": ["charme, mythology"]},
    "zapher": {
        "character": ["zapher"],
        "trigger": ["zapher, sonic the hedgehog \\(series\\)"],
    },
    "cyril_(spyro)": {
        "character": ["cyril_(spyro)"],
        "trigger": ["cyril \\(spyro\\), mythology"],
    },
    "james_(james_howard)": {
        "character": ["james_(james_howard)"],
        "trigger": ["james \\(james howard\\), patreon"],
    },
    "shello_lakoda": {
        "character": ["shello_lakoda"],
        "trigger": ["shello lakoda, disney"],
    },
    "cooler_(dragon_ball)": {
        "character": ["cooler_(dragon_ball)"],
        "trigger": ["cooler \\(dragon ball\\), dragon ball"],
    },
    "heather_kowalski": {
        "character": ["heather_kowalski"],
        "trigger": ["heather kowalski, the tishen transformation"],
    },
    "sairaks": {"character": ["sairaks"], "trigger": ["sairaks, the elder scrolls"]},
    "valkenhayn_r._hellsing": {
        "character": ["valkenhayn_r._hellsing"],
        "trigger": ["valkenhayn r. hellsing, arc system works"],
    },
    "skids": {
        "character": ["skids"],
        "trigger": ["skids, the secret lives of flowers"],
    },
    "argit": {"character": ["argit"], "trigger": ["argit, cartoon network"]},
    "sunny_(chalo)": {
        "character": ["sunny_(chalo)"],
        "trigger": ["sunny \\(chalo\\), las lindas"],
    },
    "lonestar_eberlain": {
        "character": ["lonestar_eberlain"],
        "trigger": ["lonestar eberlain, lonestarwolfoftherange"],
    },
    "goldie_o'gilt": {
        "character": ["goldie_o'gilt"],
        "trigger": ["goldie o'gilt, disney"],
    },
    "alperion": {"character": ["alperion"], "trigger": ["alperion, mythology"]},
    "kuro_(tooboe_bookmark)": {
        "character": ["kuro_(tooboe_bookmark)"],
        "trigger": ["kuro \\(tooboe bookmark\\), tooboe bookmark"],
    },
    "hermit_fox_byakudan": {
        "character": ["hermit_fox_byakudan"],
        "trigger": ["hermit fox byakudan, concon-collector"],
    },
    "akukun": {"character": ["akukun"], "trigger": ["akukun, east asian mythology"]},
    "tinker_(hladilnik)": {
        "character": ["tinker_(hladilnik)"],
        "trigger": ["tinker \\(hladilnik\\), electro-motive-diesel"],
    },
    "matt_(wackyfox26)": {
        "character": ["matt_(wackyfox26)"],
        "trigger": ["matt \\(wackyfox26\\), disney"],
    },
    "dulcinea_(puss_in_boots)": {
        "character": ["dulcinea_(puss_in_boots)"],
        "trigger": ["dulcinea \\(puss in boots\\), puss in boots \\(dreamworks\\)"],
    },
    "ray_(takemoto_arashi)": {
        "character": ["ray_(takemoto_arashi)"],
        "trigger": ["ray \\(takemoto arashi\\), lifewonders"],
    },
    "cyberblade_(character)": {
        "character": ["cyberblade_(character)"],
        "trigger": ["cyberblade \\(character\\), valentine's day"],
    },
    "liska_(scalie_schoolie)": {
        "character": ["liska_(scalie_schoolie)"],
        "trigger": ["liska \\(scalie schoolie\\), scalie schoolie"],
    },
    "owen_(repeat)": {
        "character": ["owen_(repeat)"],
        "trigger": ["owen \\(repeat\\), repeat \\(visual novel\\)"],
    },
    "tuki_(shantae)": {
        "character": ["tuki_(shantae)"],
        "trigger": ["tuki \\(shantae\\), wayforward"],
    },
    "leskaviene": {"character": ["leskaviene"], "trigger": ["leskaviene, nintendo"]},
    "stuart_(naughtymorg)": {
        "character": ["stuart_(naughtymorg)"],
        "trigger": ["stuart \\(naughtymorg\\), mythology"],
    },
    "wicke_(pokemon)": {
        "character": ["wicke_(pokemon)"],
        "trigger": ["wicke \\(pokemon\\), pokemon"],
    },
    "balros_(echoen)": {
        "character": ["balros_(echoen)"],
        "trigger": ["balros \\(echoen\\), jade ankh"],
    },
    "devil_teemo_(lol)": {
        "character": ["devil_teemo_(lol)"],
        "trigger": ["devil teemo \\(lol\\), riot games"],
    },
    "pandora's_fox": {
        "character": ["pandora's_fox"],
        "trigger": ["pandora's fox, mythology"],
    },
    "rick_marks": {"character": ["rick_marks"], "trigger": ["rick marks, mythology"]},
    "hangetsu_(ko-gami)": {
        "character": ["hangetsu_(ko-gami)"],
        "trigger": ["hangetsu \\(ko-gami\\), christmas"],
    },
    "pascal_(cudacore)": {
        "character": ["pascal_(cudacore)"],
        "trigger": ["pascal \\(cudacore\\), disney"],
    },
    "liquiir": {"character": ["liquiir"], "trigger": ["liquiir, dragon ball"]},
    "bimm": {"character": ["bimm"], "trigger": ["bimm, mighty magiswords"]},
    "yamia_(lunaflame)": {
        "character": ["yamia_(lunaflame)"],
        "trigger": ["yamia \\(lunaflame\\), natura \\(lunaflame\\)"],
    },
    "amy_(lcut)": {
        "character": ["amy_(lcut)"],
        "trigger": ["amy \\(lcut\\), mythology"],
    },
    "scp-811": {"character": ["scp-811"], "trigger": ["scp-811, scp foundation"]},
    "kobe_bear": {"character": ["kobe_bear"], "trigger": ["kobe bear, pokemon"]},
    "caring_hearts_(mlp)": {
        "character": ["caring_hearts_(mlp)"],
        "trigger": ["caring hearts \\(mlp\\), my little pony"],
    },
    "rocky_(kusosensei)": {
        "character": ["rocky_(kusosensei)"],
        "trigger": ["rocky \\(kusosensei\\), valentine's day"],
    },
    "rimuru_tempest": {
        "character": ["rimuru_tempest"],
        "trigger": ["rimuru tempest, that time i got reincarnated as a slime"],
    },
    "doug_(101_dalmatians)": {
        "character": ["doug_(101_dalmatians)"],
        "trigger": ["doug \\(101 dalmatians\\), disney"],
    },
    "toy_chica_(eroticphobia)": {
        "character": ["toy_chica_(eroticphobia)"],
        "trigger": ["toy chica \\(eroticphobia\\), scottgames"],
    },
    "mattie_(pokefound)": {
        "character": ["mattie_(pokefound)"],
        "trigger": ["mattie \\(pokefound\\), da silva"],
    },
    "neve_vecat": {"character": ["neve_vecat"], "trigger": ["neve vecat, mythology"]},
    "roxanne_(spikedmauler)": {
        "character": ["roxanne_(spikedmauler)"],
        "trigger": ["roxanne \\(spikedmauler\\), creative commons"],
    },
    "ryme_(totodice1)": {
        "character": ["ryme_(totodice1)"],
        "trigger": ["ryme \\(totodice1\\), pokemon"],
    },
    "hilwu": {"character": ["hilwu"], "trigger": ["hilwu, blender \\(software\\)"]},
    "virgil_(funkspunky)": {
        "character": ["virgil_(funkspunky)"],
        "trigger": ["virgil \\(funkspunky\\), mythology"],
    },
    "tobias_(thehades)": {
        "character": ["tobias_(thehades)"],
        "trigger": ["tobias \\(thehades\\), mythology"],
    },
    "asterion_(minotaur_hotel)": {
        "character": ["asterion_(minotaur_hotel)"],
        "trigger": ["asterion \\(minotaur hotel\\), minotaur hotel"],
    },
    "byte_fantail_(character)": {
        "character": ["byte_fantail_(character)"],
        "trigger": ["byte fantail \\(character\\), mythology"],
    },
    "ninomae_ina'nis": {
        "character": ["ninomae_ina'nis"],
        "trigger": ["ninomae ina'nis, hololive"],
    },
    "kalebur": {"character": ["kalebur"], "trigger": ["kalebur, pokemon"]},
    "quill_wonderfowl": {
        "character": ["quill_wonderfowl"],
        "trigger": ["quill wonderfowl, disney"],
    },
    "ciaran_(tailmaw)": {
        "character": ["ciaran_(tailmaw)"],
        "trigger": ["ciaran \\(tailmaw\\), pokemon"],
    },
    "jat_(thepatchedragon)": {
        "character": ["jat_(thepatchedragon)"],
        "trigger": ["jat \\(thepatchedragon\\), mythology"],
    },
    "ashley_(mutagen)": {
        "character": ["ashley_(mutagen)"],
        "trigger": ["ashley \\(mutagen\\), mythology"],
    },
    "klank_(spark.kobbo)": {
        "character": ["klank_(spark.kobbo)"],
        "trigger": ["klank \\(spark.kobbo\\), mythology"],
    },
    "scarlet_(coel3d)": {
        "character": ["scarlet_(coel3d)"],
        "trigger": ["scarlet \\(coel3d\\), sonic the hedgehog \\(series\\)"],
    },
    "topper_(nu:_carnival)": {
        "character": ["topper_(nu:_carnival)"],
        "trigger": ["topper \\(nu: carnival\\), nu: carnival"],
    },
    "lily_(atrumpet)": {
        "character": ["lily_(atrumpet)"],
        "trigger": ["lily \\(atrumpet\\), mythology"],
    },
    "tsukiko_(rekidesu)": {
        "character": ["tsukiko_(rekidesu)"],
        "trigger": ["tsukiko \\(rekidesu\\), mythology"],
    },
    "arokha": {"character": ["arokha"], "trigger": ["arokha, okami \\(capcom\\)"]},
    "jolly_jack_(character)": {
        "character": ["jolly_jack_(character)"],
        "trigger": ["jolly jack \\(character\\), mythology"],
    },
    "croix": {"character": ["croix"], "trigger": ["croix, nintendo"]},
    "buck_(buckdragon)": {
        "character": ["buck_(buckdragon)"],
        "trigger": ["buck \\(buckdragon\\), mythology"],
    },
    "krystal_(dinosaur_planet)": {
        "character": ["krystal_(dinosaur_planet)"],
        "trigger": ["krystal \\(dinosaur planet\\), dinosaur planet"],
    },
    "henry_wong": {"character": ["henry_wong"], "trigger": ["henry wong, digimon"]},
    "cirno": {"character": ["cirno"], "trigger": ["cirno, touhou"]},
    "killer_croc": {
        "character": ["killer_croc"],
        "trigger": ["killer croc, dc comics"],
    },
    "secret_squirrel": {
        "character": ["secret_squirrel"],
        "trigger": ["secret squirrel, hanna-barbera"],
    },
    "shen": {"character": ["shen"], "trigger": ["shen, mythology"]},
    "penelope_(sly_cooper)": {
        "character": ["penelope_(sly_cooper)"],
        "trigger": ["penelope \\(sly cooper\\), sucker punch productions"],
    },
    "batgirl": {"character": ["batgirl"], "trigger": ["batgirl, dc comics"]},
    "brain_(top_cat)": {
        "character": ["brain_(top_cat)"],
        "trigger": ["brain \\(top cat\\), top cat \\(series\\)"],
    },
    "deihnyx": {"character": ["deihnyx"], "trigger": ["deihnyx, mythology"]},
    "aunt_orange_(mlp)": {
        "character": ["aunt_orange_(mlp)"],
        "trigger": ["aunt orange \\(mlp\\), my little pony"],
    },
    "joanna_watterson": {
        "character": ["joanna_watterson"],
        "trigger": ["joanna watterson, cartoon network"],
    },
    "sir_percival_(sonic_and_the_black_knight)": {
        "character": ["sir_percival_(sonic_and_the_black_knight)"],
        "trigger": [
            "sir percival \\(sonic and the black knight\\), sonic storybook series"
        ],
    },
    "abby_sinian": {
        "character": ["abby_sinian"],
        "trigger": ["abby sinian, swat kats"],
    },
    "pusheen": {"character": ["pusheen"], "trigger": ["pusheen, pusheen and friends"]},
    "pun_pony": {"character": ["pun_pony"], "trigger": ["pun pony, my little pony"]},
    "pinky_(pac-man)": {
        "character": ["pinky_(pac-man)"],
        "trigger": ["pinky \\(pac-man\\), pac-man \\(series\\)"],
    },
    "nate_(pokemon)": {
        "character": ["nate_(pokemon)"],
        "trigger": ["nate \\(pokemon\\), pokemon"],
    },
    "nicky_(thea_sisters)": {
        "character": ["nicky_(thea_sisters)"],
        "trigger": ["nicky \\(thea sisters\\), geronimo stilton \\(series\\)"],
    },
    "chuukichi_(morenatsu)": {
        "character": ["chuukichi_(morenatsu)"],
        "trigger": ["chuukichi \\(morenatsu\\), morenatsu"],
    },
    "blue_dragon_(character)": {
        "character": ["blue_dragon_(character)"],
        "trigger": ["blue dragon \\(character\\), mythology"],
    },
    "frost_(cinderfrost)": {
        "character": ["frost_(cinderfrost)"],
        "trigger": ["frost \\(cinderfrost\\), cinderfrost"],
    },
    "neelix_(character)": {
        "character": ["neelix_(character)"],
        "trigger": ["neelix \\(character\\), guardians of the galaxy"],
    },
    "huttser-coyote_(character)": {
        "character": ["huttser-coyote_(character)"],
        "trigger": ["huttser-coyote \\(character\\), mythology"],
    },
    "tanya_(mcnasty)": {
        "character": ["tanya_(mcnasty)"],
        "trigger": ["tanya \\(mcnasty\\), pokemon"],
    },
    "rabbit_(wolfpack67)": {
        "character": ["rabbit_(wolfpack67)"],
        "trigger": ["rabbit \\(wolfpack67\\), wolfpack67"],
    },
    "paralee_(character)": {
        "character": ["paralee_(character)"],
        "trigger": ["paralee \\(character\\), warcraft"],
    },
    "apple_jewel_(mlp)": {
        "character": ["apple_jewel_(mlp)"],
        "trigger": ["apple jewel \\(mlp\\), my little pony"],
    },
    "epsilon": {"character": ["epsilon"], "trigger": ["epsilon, mythology"]},
    "snap_feather": {
        "character": ["snap_feather"],
        "trigger": ["snap feather, my little pony"],
    },
    "dessy": {"character": ["dessy"], "trigger": ["dessy, mythology"]},
    "matoi-chan_(mamoru-kun)": {
        "character": ["matoi-chan_(mamoru-kun)"],
        "trigger": ["matoi-chan \\(mamoru-kun\\), little tail bronx"],
    },
    "topaz_lareme_(battler)": {
        "character": ["topaz_lareme_(battler)"],
        "trigger": ["topaz lareme \\(battler\\), cub con"],
    },
    "angle_(copperback01)": {
        "character": ["angle_(copperback01)"],
        "trigger": ["angle \\(copperback01\\), christmas"],
    },
    "reinhardt_(overwatch)": {
        "character": ["reinhardt_(overwatch)"],
        "trigger": ["reinhardt \\(overwatch\\), blizzard entertainment"],
    },
    "radwolf": {"character": ["radwolf"], "trigger": ["radwolf, jakescorp"]},
    "harzipan": {"character": ["harzipan"], "trigger": ["harzipan, pokemon"]},
    "milenth_drake": {
        "character": ["milenth_drake"],
        "trigger": ["milenth drake, mythology"],
    },
    "dean_(terribleanimal)": {
        "character": ["dean_(terribleanimal)"],
        "trigger": ["dean \\(terribleanimal\\), animal crossing"],
    },
    "aya_blackpaw": {
        "character": ["aya_blackpaw"],
        "trigger": ["aya blackpaw, hearthstone"],
    },
    "yugia_(evov1)": {
        "character": ["yugia_(evov1)"],
        "trigger": ["yugia \\(evov1\\), pokemon"],
    },
    "sugaryhotdog_(character)": {
        "character": ["sugaryhotdog_(character)"],
        "trigger": ["sugaryhotdog \\(character\\), mythology"],
    },
    "ruchex_(character)": {
        "character": ["ruchex_(character)"],
        "trigger": ["ruchex \\(character\\), pokemon"],
    },
    "coffeesoda_(fursona)": {
        "character": ["coffeesoda_(fursona)"],
        "trigger": ["coffeesoda \\(fursona\\), mythology"],
    },
    "dook_(lildooks)": {
        "character": ["dook_(lildooks)"],
        "trigger": ["dook \\(lildooks\\), my little pony"],
    },
    "lucas_(lucasreturns)": {
        "character": ["lucas_(lucasreturns)"],
        "trigger": ["lucas \\(lucasreturns\\), mythology"],
    },
    "maple_(animal_crossing)": {
        "character": ["maple_(animal_crossing)"],
        "trigger": ["maple \\(animal crossing\\), animal crossing"],
    },
    "patt_(waver-ring)": {
        "character": ["patt_(waver-ring)"],
        "trigger": ["patt \\(waver-ring\\), illumination entertainment"],
    },
    "jaren_(foxyrexy)": {
        "character": ["jaren_(foxyrexy)"],
        "trigger": ["jaren \\(foxyrexy\\), mythology"],
    },
    "wolfthorn_(old_spice)": {
        "character": ["wolfthorn_(old_spice)"],
        "trigger": ["wolfthorn \\(old spice\\), old spice"],
    },
    "regina_(ragnacock)": {
        "character": ["regina_(ragnacock)"],
        "trigger": ["regina \\(ragnacock\\), christmas"],
    },
    "nate_(littlerager)": {
        "character": ["nate_(littlerager)"],
        "trigger": ["nate \\(littlerager\\), pokemon"],
    },
    "puppycorn": {"character": ["puppycorn"], "trigger": ["puppycorn, lego"]},
    "rough_the_skunk": {
        "character": ["rough_the_skunk"],
        "trigger": ["rough the skunk, sonic the hedgehog \\(series\\)"],
    },
    "valentina_(aimbot-jones)": {
        "character": ["valentina_(aimbot-jones)"],
        "trigger": ["valentina \\(aimbot-jones\\), pokemon"],
    },
    "daven_(dado463art)": {
        "character": ["daven_(dado463art)"],
        "trigger": ["daven \\(dado463art\\), neon genesis evangelion"],
    },
    "miu_(aas)": {"character": ["miu_(aas)"], "trigger": ["miu \\(aas\\), pocky"]},
    "lovers_(oc)": {
        "character": ["lovers_(oc)"],
        "trigger": ["lovers \\(oc\\), my little pony"],
    },
    "annoy_(character)": {
        "character": ["annoy_(character)"],
        "trigger": ["annoy \\(character\\), kinktober"],
    },
    "pepper_(halbean)": {
        "character": ["pepper_(halbean)"],
        "trigger": ["pepper \\(halbean\\), christmas"],
    },
    "dude_lyena": {"character": ["dude_lyena"], "trigger": ["dude lyena, halloween"]},
    "helelos": {"character": ["helelos"], "trigger": ["helelos, escape to nowhere"]},
    "alys_faiblesse_(zelripheth)": {
        "character": ["alys_faiblesse_(zelripheth)"],
        "trigger": ["alys faiblesse \\(zelripheth\\), christmas"],
    },
    "liz_(lizzycat21)": {
        "character": ["liz_(lizzycat21)"],
        "trigger": ["liz \\(lizzycat21\\), pokemon"],
    },
    "chloe_(glopossum)": {
        "character": ["chloe_(glopossum)"],
        "trigger": ["chloe \\(glopossum\\), mythology"],
    },
    "stan_the_woozle": {
        "character": ["stan_the_woozle"],
        "trigger": ["stan the woozle, disney"],
    },
    "luck_(icma)": {
        "character": ["luck_(icma)"],
        "trigger": ["luck \\(icma\\), pokemon"],
    },
    "roxie_(frozenartifice)": {
        "character": ["roxie_(frozenartifice)"],
        "trigger": ["roxie \\(frozenartifice\\), no nut november"],
    },
    "spanx": {"character": ["spanx"], "trigger": ["spanx, whiplash \\(game\\)"]},
    "rangstrom": {"character": ["rangstrom"], "trigger": ["rangstrom, mythology"]},
    "guilmon_(furromantic)": {
        "character": ["guilmon_(furromantic)"],
        "trigger": ["guilmon \\(furromantic\\), digimon"],
    },
    "penta_(cum.cat)": {
        "character": ["penta_(cum.cat)"],
        "trigger": ["penta \\(cum.cat\\), mythology"],
    },
    "chip_(catmakinbiscuits)": {
        "character": ["chip_(catmakinbiscuits)"],
        "trigger": ["chip \\(catmakinbiscuits\\), mythology"],
    },
    "miryam_(giru)": {
        "character": ["miryam_(giru)"],
        "trigger": ["miryam \\(giru\\), mythology"],
    },
    "babs_(snoot_game)": {
        "character": ["babs_(snoot_game)"],
        "trigger": ["babs \\(snoot game\\), cavemanon studios"],
    },
    "red_(sluggabed)": {
        "character": ["red_(sluggabed)"],
        "trigger": ["red \\(sluggabed\\), mythology"],
    },
    "mama_bear": {
        "character": ["mama_bear"],
        "trigger": ["mama bear, berenstain bears"],
    },
    "thorn": {"character": ["thorn"], "trigger": ["thorn, mythology"]},
    "astaroth": {"character": ["astaroth"], "trigger": ["astaroth, shinrabanshou"]},
    "arthur_mathews": {
        "character": ["arthur_mathews"],
        "trigger": ["arthur mathews, sequential art"],
    },
    "clair_(pokemon)": {
        "character": ["clair_(pokemon)"],
        "trigger": ["clair \\(pokemon\\), pokemon"],
    },
    "lisa_(goof_troop)": {
        "character": ["lisa_(goof_troop)"],
        "trigger": ["lisa \\(goof troop\\), disney"],
    },
    "lucy_hare": {"character": ["lucy_hare"], "trigger": ["lucy hare, nintendo"]},
    "ace_bunny": {
        "character": ["ace_bunny"],
        "trigger": ["ace bunny, loonatics unleashed"],
    },
    "bob_(bubble_bobble)": {
        "character": ["bob_(bubble_bobble)"],
        "trigger": ["bob \\(bubble bobble\\), bubble bobble"],
    },
    "kimberly_ann_possible": {
        "character": ["kimberly_ann_possible"],
        "trigger": ["kimberly ann possible, kim possible"],
    },
    "bub_(bubble_bobble)": {
        "character": ["bub_(bubble_bobble)"],
        "trigger": ["bub \\(bubble bobble\\), taito"],
    },
    "sandy_(bcb)": {
        "character": ["sandy_(bcb)"],
        "trigger": ["sandy \\(bcb\\), bittersweet candy bowl"],
    },
    "pegasi_guard_(mlp)": {
        "character": ["pegasi_guard_(mlp)"],
        "trigger": ["pegasi guard \\(mlp\\), my little pony"],
    },
    "rouge_the_werebat": {
        "character": ["rouge_the_werebat"],
        "trigger": ["rouge the werebat, sonic the hedgehog \\(series\\)"],
    },
    "utunu": {"character": ["utunu"], "trigger": ["utunu, egyptian mythology"]},
    "skoop": {"character": ["skoop"], "trigger": ["skoop, mythology"]},
    "dillon_(dillon's_rolling_western)": {
        "character": ["dillon_(dillon's_rolling_western)"],
        "trigger": ["dillon \\(dillon's rolling western\\), dillon's rolling western"],
    },
    "fuleco": {"character": ["fuleco"], "trigger": ["fuleco, fifa"]},
    "roland_guiscard": {
        "character": ["roland_guiscard"],
        "trigger": ["roland guiscard, the adventures of tintin"],
    },
    "blizzie_(blizzieart)": {
        "character": ["blizzie_(blizzieart)"],
        "trigger": ["blizzie \\(blizzieart\\), mythology"],
    },
    "patricia_wagon": {
        "character": ["patricia_wagon"],
        "trigger": ["patricia wagon, mighty switch force!"],
    },
    "deviltod": {"character": ["deviltod"], "trigger": ["deviltod, pokemon"]},
    "carrot_(carrot)": {
        "character": ["carrot_(carrot)"],
        "trigger": ["carrot \\(carrot\\), easter"],
    },
    "chu_(duckdraw)": {
        "character": ["chu_(duckdraw)"],
        "trigger": ["chu \\(duckdraw\\), pokemon"],
    },
    "ethan_(teckly)": {
        "character": ["ethan_(teckly)"],
        "trigger": ["ethan \\(teckly\\), mythology"],
    },
    "arisu_starfall": {
        "character": ["arisu_starfall"],
        "trigger": ["arisu starfall, mythology"],
    },
    "sarah_van_fiepland": {
        "character": ["sarah_van_fiepland"],
        "trigger": ["sarah van fiepland, mythology"],
    },
    "tammy_(animal_crossing)": {
        "character": ["tammy_(animal_crossing)"],
        "trigger": ["tammy \\(animal crossing\\), animal crossing"],
    },
    "mira_(animal_crossing)": {
        "character": ["mira_(animal_crossing)"],
        "trigger": ["mira \\(animal crossing\\), animal crossing"],
    },
    "cobaltbadger_(character)": {
        "character": ["cobaltbadger_(character)"],
        "trigger": ["cobaltbadger \\(character\\), guardians of the galaxy"],
    },
    "lucia_(grey_wolf_570)": {
        "character": ["lucia_(grey_wolf_570)"],
        "trigger": ["lucia \\(grey wolf 570\\)"],
    },
    "chi_chi": {"character": ["chi_chi"], "trigger": ["chi chi, cartoon network"]},
    "zake": {"character": ["zake"], "trigger": ["zake, bandai namco"]},
    "irah_(fvt)": {
        "character": ["irah_(fvt)"],
        "trigger": ["irah \\(fvt\\), fairies vs tentacles"],
    },
    "jet_(quin_nsfw)": {
        "character": ["jet_(quin_nsfw)"],
        "trigger": ["jet \\(quin nsfw\\), nintendo"],
    },
    "rothfale": {"character": ["rothfale"], "trigger": ["rothfale, mythology"]},
    "casy_the_wolfcat": {
        "character": ["casy_the_wolfcat"],
        "trigger": ["casy the wolfcat, dogshaming"],
    },
    "spencer_(lonewolfhowling)": {
        "character": ["spencer_(lonewolfhowling)"],
        "trigger": ["spencer \\(lonewolfhowling\\), sony interactive entertainment"],
    },
    "raptoral_(character)": {
        "character": ["raptoral_(character)"],
        "trigger": ["raptoral \\(character\\), mythology"],
    },
    "felicia_(brushfire)": {
        "character": ["felicia_(brushfire)"],
        "trigger": ["felicia \\(brushfire\\), mythology"],
    },
    "tony_(terribleanimal)": {
        "character": ["tony_(terribleanimal)"],
        "trigger": ["tony \\(terribleanimal\\), animal crossing"],
    },
    "buck_(brushfire)": {
        "character": ["buck_(brushfire)"],
        "trigger": ["buck \\(brushfire\\), mythology"],
    },
    "west_of_heaven": {
        "character": ["west_of_heaven"],
        "trigger": ["west of heaven, ah club"],
    },
    "bael_thunderfist": {
        "character": ["bael_thunderfist"],
        "trigger": ["bael thunderfist, warcraft"],
    },
    "hammerface": {"character": ["hammerface"], "trigger": ["hammerface, disney"]},
    "scrap_baby_(fnaf)": {
        "character": ["scrap_baby_(fnaf)"],
        "trigger": ["scrap baby \\(fnaf\\), scottgames"],
    },
    "parappa_the_trappa": {
        "character": ["parappa_the_trappa"],
        "trigger": ["parappa the trappa, parappa the rapper"],
    },
    "nyn_indigo": {
        "character": ["nyn_indigo"],
        "trigger": ["nyn indigo, my little pony"],
    },
    "baz_badger": {"character": ["baz_badger"], "trigger": ["baz badger, nintendo"]},
    "eliana_corvalis": {
        "character": ["eliana_corvalis"],
        "trigger": ["eliana corvalis, mass effect"],
    },
    "mitsuhide_vulpes": {
        "character": ["mitsuhide_vulpes"],
        "trigger": ["mitsuhide vulpes, mythology"],
    },
    "destiny_(101_dalmatians)": {
        "character": ["destiny_(101_dalmatians)"],
        "trigger": ["destiny \\(101 dalmatians\\), disney"],
    },
    "ruka_vaporeon": {
        "character": ["ruka_vaporeon"],
        "trigger": ["ruka vaporeon, pokemon"],
    },
    "khanivore": {
        "character": ["khanivore"],
        "trigger": ["khanivore, love death + robots"],
    },
    "lemy": {"character": ["lemy"], "trigger": ["lemy, mythology"]},
    "len_(focus)": {
        "character": ["len_(focus)"],
        "trigger": ["len \\(focus\\), pokemon"],
    },
    "hill_(father_hill)": {
        "character": ["hill_(father_hill)"],
        "trigger": ["hill \\(father hill\\), owo whats this"],
    },
    "zan_(zantanerz)": {
        "character": ["zan_(zantanerz)"],
        "trigger": ["zan \\(zantanerz\\), nintendo"],
    },
    "khander": {"character": ["khander"], "trigger": ["khander, mythology"]},
    "lindsay_(funkybun)": {
        "character": ["lindsay_(funkybun)"],
        "trigger": ["lindsay \\(funkybun\\), patreon"],
    },
    "smiley_cindy_(skashi95)": {
        "character": ["smiley_cindy_(skashi95)"],
        "trigger": ["smiley cindy \\(skashi95\\), little laughters"],
    },
    "squint_(leobo)": {
        "character": ["squint_(leobo)"],
        "trigger": ["squint \\(leobo\\), patreon"],
    },
    "fivey_fox": {
        "character": ["fivey_fox"],
        "trigger": ["fivey fox, fivethirtyeight"],
    },
    "orianne_larone": {
        "character": ["orianne_larone"],
        "trigger": ["orianne larone, mythology"],
    },
    "gavin_pearson": {
        "character": ["gavin_pearson"],
        "trigger": ["gavin pearson, joy division \\(band\\)"],
    },
    "feruda_(farstaria)": {
        "character": ["feruda_(farstaria)"],
        "trigger": ["feruda \\(farstaria\\), mythology"],
    },
    "jeremy_(topazknight)": {
        "character": ["jeremy_(topazknight)"],
        "trigger": ["jeremy \\(topazknight\\), mythology"],
    },
    "rook_(lazymoose)": {
        "character": ["rook_(lazymoose)"],
        "trigger": ["rook \\(lazymoose\\), halloween"],
    },
    "lua_(pacothegint)": {
        "character": ["lua_(pacothegint)"],
        "trigger": ["lua \\(pacothegint\\), mythology"],
    },
    "antay_(hevinsane)": {
        "character": ["antay_(hevinsane)"],
        "trigger": ["antay \\(hevinsane\\), mythology"],
    },
    "huggy_wuggy": {
        "character": ["huggy_wuggy"],
        "trigger": ["huggy wuggy, poppy playtime"],
    },
    "sander_(dislyte)": {
        "character": ["sander_(dislyte)"],
        "trigger": ["sander \\(dislyte\\), dislyte"],
    },
    "contractor": {"character": ["contractor"], "trigger": ["contractor, vtuber"]},
    "rufus_(fortnite)": {
        "character": ["rufus_(fortnite)"],
        "trigger": ["rufus \\(fortnite\\), fortnite"],
    },
    "chessly": {"character": ["chessly"], "trigger": ["chessly, mythology"]},
    "vide_(bjekkergauken)": {
        "character": ["vide_(bjekkergauken)"],
        "trigger": ["vide \\(bjekkergauken\\)"],
    },
    "sunni_smiles": {
        "character": ["sunni_smiles"],
        "trigger": ["sunni smiles, sonic the hedgehog \\(series\\)"],
    },
    "yuffie_kisaragi": {
        "character": ["yuffie_kisaragi"],
        "trigger": ["yuffie kisaragi, square enix"],
    },
    "sparky_the_chu_(character)": {
        "character": ["sparky_the_chu_(character)"],
        "trigger": ["sparky the chu \\(character\\), pokemon"],
    },
    "francine_manx": {
        "character": ["francine_manx"],
        "trigger": ["francine manx, samurai pizza cats"],
    },
    "inuki_(character)": {
        "character": ["inuki_(character)"],
        "trigger": ["inuki \\(character\\), pokemon"],
    },
    "ember_(elitetheespeon)": {
        "character": ["ember_(elitetheespeon)"],
        "trigger": ["ember \\(elitetheespeon\\), pokemon"],
    },
    "mertle_edmonds": {
        "character": ["mertle_edmonds"],
        "trigger": ["mertle edmonds, disney"],
    },
    "chomper_(the_land_before_time)": {
        "character": ["chomper_(the_land_before_time)"],
        "trigger": ["chomper \\(the land before time\\), don bluth"],
    },
    "bomba_(krillos)": {
        "character": ["bomba_(krillos)"],
        "trigger": ["bomba \\(krillos\\), mythology"],
    },
    "viktor_vasko": {
        "character": ["viktor_vasko"],
        "trigger": ["viktor vasko, lackadaisy"],
    },
    "chuck_(braford)": {
        "character": ["chuck_(braford)"],
        "trigger": ["chuck \\(braford\\), house of beef"],
    },
    "diamondhead": {
        "character": ["diamondhead"],
        "trigger": ["diamondhead, cartoon network"],
    },
    "mara_(gunmouth)": {
        "character": ["mara_(gunmouth)"],
        "trigger": ["mara \\(gunmouth\\), clubstripes"],
    },
    "vixenchan": {"character": ["vixenchan"], "trigger": ["vixenchan, vixen defea"]},
    "moo_lawgoat": {
        "character": ["moo_lawgoat"],
        "trigger": ["moo lawgoat, mythology"],
    },
    "aurelina_canidae": {
        "character": ["aurelina_canidae"],
        "trigger": ["aurelina canidae, disney"],
    },
    "kaltag_(balto)": {
        "character": ["kaltag_(balto)"],
        "trigger": ["kaltag \\(balto\\), universal studios"],
    },
    "jak": {"character": ["jak"], "trigger": ["jak, jak and daxter"]},
    "sparks_pichu": {
        "character": ["sparks_pichu"],
        "trigger": ["sparks pichu, pokemon"],
    },
    "shaak_ti": {"character": ["shaak_ti"], "trigger": ["shaak ti, star wars"]},
    "david_(bcb)": {
        "character": ["david_(bcb)"],
        "trigger": ["david \\(bcb\\), bittersweet candy bowl"],
    },
    "red_(angry_birds)": {
        "character": ["red_(angry_birds)"],
        "trigger": ["red \\(angry birds\\), rovio entertainment"],
    },
    "thanos": {"character": ["thanos"], "trigger": ["thanos, marvel"]},
    "shorttail": {"character": ["shorttail"], "trigger": ["shorttail, mythology"]},
    "minx_kitten": {"character": ["minx_kitten"], "trigger": ["minx kitten, nike"]},
    "rafiki": {"character": ["rafiki"], "trigger": ["rafiki, disney"]},
    "francine_frensky": {
        "character": ["francine_frensky"],
        "trigger": ["francine frensky, arthur \\(series\\)"],
    },
    "seth_häser": {"character": ["seth_häser"], "trigger": ["seth häser, fallout"]},
    "imperfect_cell": {
        "character": ["imperfect_cell"],
        "trigger": ["imperfect cell, dragon ball"],
    },
    "velux": {"character": ["velux"], "trigger": ["velux, mythology"]},
    "dian_(jewelpet)": {
        "character": ["dian_(jewelpet)"],
        "trigger": ["dian \\(jewelpet\\), jewelpet"],
    },
    "kenta_shiba_(character)": {
        "character": ["kenta_shiba_(character)"],
        "trigger": ["kenta shiba \\(character\\), pocky and pretz day"],
    },
    "selicia": {"character": ["selicia"], "trigger": ["selicia, mythology"]},
    "tildriel": {"character": ["tildriel"], "trigger": ["tildriel, tale of tails"]},
    "kyorg7": {"character": ["kyorg7"], "trigger": ["kyorg7, mythology"]},
    "shrike_alvaron": {
        "character": ["shrike_alvaron"],
        "trigger": ["shrike alvaron, pokemon"],
    },
    "airalin": {"character": ["airalin"], "trigger": ["airalin, pokemon"]},
    "bonbon_(roommates)": {
        "character": ["bonbon_(roommates)"],
        "trigger": ["bonbon \\(roommates\\), roommates:motha"],
    },
    "set_(puzzle_and_dragons)": {
        "character": ["set_(puzzle_and_dragons)"],
        "trigger": ["set \\(puzzle and dragons\\), puzzle and dragons"],
    },
    "riku_tavash": {"character": ["riku_tavash"], "trigger": ["riku tavash, cyberia"]},
    "greg_universe": {
        "character": ["greg_universe"],
        "trigger": ["greg universe, cartoon network"],
    },
    "takemoto": {"character": ["takemoto"], "trigger": ["takemoto, lifewonders"]},
    "ludwig_bullworth_jackson": {
        "character": ["ludwig_bullworth_jackson"],
        "trigger": [
            "ludwig bullworth jackson, ludwig bullworth jackson \\(copyright\\)"
        ],
    },
    "dracarna": {"character": ["dracarna"], "trigger": ["dracarna, mythology"]},
    "cookie_dough_(oc)": {
        "character": ["cookie_dough_(oc)"],
        "trigger": ["cookie dough \\(oc\\), christmas"],
    },
    "pixie_blume": {"character": ["pixie_blume"], "trigger": ["pixie blume, pokemon"]},
    "araiguma-san": {
        "character": ["araiguma-san"],
        "trigger": ["araiguma-san, arc system works"],
    },
    "randy_rabbit": {
        "character": ["randy_rabbit"],
        "trigger": ["randy rabbit, rutwell forest"],
    },
    "deborah_bispo": {
        "character": ["deborah_bispo"],
        "trigger": ["deborah bispo, jonas brasileiro \\(copyright\\)"],
    },
    "ember_(angstrom)": {
        "character": ["ember_(angstrom)"],
        "trigger": ["ember \\(angstrom\\), undertale \\(series\\)"],
    },
    "wang_chow": {
        "character": ["wang_chow"],
        "trigger": ["wang chow, gab \\(comic\\)"],
    },
    "bella_(screwroot)": {
        "character": ["bella_(screwroot)"],
        "trigger": ["bella \\(screwroot\\), mythology"],
    },
    "mungo_(housepets!)": {
        "character": ["mungo_(housepets!)"],
        "trigger": ["mungo \\(housepets!\\), housepets!"],
    },
    "jenn_(irkingir)": {
        "character": ["jenn_(irkingir)"],
        "trigger": ["jenn \\(irkingir\\), mythology"],
    },
    "tenshi_chan": {"character": ["tenshi_chan"], "trigger": ["tenshi chan, easter"]},
    "adian_(moki)": {
        "character": ["adian_(moki)"],
        "trigger": ["adian \\(moki\\), halloween"],
    },
    "jack_dragon_(character)": {
        "character": ["jack_dragon_(character)"],
        "trigger": ["jack dragon \\(character\\), mythology"],
    },
    "teslawolfen": {
        "character": ["teslawolfen"],
        "trigger": ["teslawolfen, mythology"],
    },
    "mocha_(mochalattefox)": {
        "character": ["mocha_(mochalattefox)"],
        "trigger": ["mocha \\(mochalattefox\\), mythology"],
    },
    "knedit": {"character": ["knedit"], "trigger": ["knedit, mythology"]},
    "lana's_mother": {
        "character": ["lana's_mother"],
        "trigger": ["lana's mother, pokemon"],
    },
    "nanahoshi_suzu": {
        "character": ["nanahoshi_suzu"],
        "trigger": ["nanahoshi suzu, vtuber"],
    },
    "yafya_(beastars)": {
        "character": ["yafya_(beastars)"],
        "trigger": ["yafya \\(beastars\\), beastars"],
    },
    "ugly_sonic": {
        "character": ["ugly_sonic"],
        "trigger": ["ugly sonic, sonic the hedgehog \\(series\\)"],
    },
    "kiro_(warcraft)": {
        "character": ["kiro_(warcraft)"],
        "trigger": ["kiro \\(warcraft\\), warcraft"],
    },
    "emmy_dook": {"character": ["emmy_dook"], "trigger": ["emmy dook, tetris"]},
    "sentai_rabbit_(marimo)": {
        "character": ["sentai_rabbit_(marimo)"],
        "trigger": ["sentai rabbit \\(marimo\\), east asian mythology"],
    },
    "nicole_(foxnick12)": {
        "character": ["nicole_(foxnick12)"],
        "trigger": ["nicole \\(foxnick12\\), mythology"],
    },
    "galathea": {"character": ["galathea"], "trigger": ["galathea, star wars"]},
    "lute_(hazbin_hotel)": {
        "character": ["lute_(hazbin_hotel)"],
        "trigger": ["lute \\(hazbin hotel\\), hazbin hotel"],
    },
    "kiera_(shot_one)": {
        "character": ["kiera_(shot_one)"],
        "trigger": ["kiera \\(shot one\\), shot one"],
    },
    "batty_(100_percent_wolf)": {
        "character": ["batty_(100_percent_wolf)"],
        "trigger": ["batty \\(100 percent wolf\\), 100 percent wolf"],
    },
    "amber_(hallogreen)": {
        "character": ["amber_(hallogreen)"],
        "trigger": ["amber \\(hallogreen\\), christmas"],
    },
    "damian_(funkybun)": {
        "character": ["damian_(funkybun)"],
        "trigger": ["damian \\(funkybun\\), patreon"],
    },
    "trill_(not_a_spider)": {
        "character": ["trill_(not_a_spider)"],
        "trigger": ["trill \\(not a spider\\), my little pony"],
    },
    "rose_(kamikazekit)": {
        "character": ["rose_(kamikazekit)"],
        "trigger": ["rose \\(kamikazekit\\), mythology"],
    },
    "thrar'ixauth": {
        "character": ["thrar'ixauth"],
        "trigger": ["thrar'ixauth, mythology"],
    },
    "robyn_goodfellowe": {
        "character": ["robyn_goodfellowe"],
        "trigger": ["robyn goodfellowe, wolfwalkers"],
    },
    "nix_(ceehaz)": {
        "character": ["nix_(ceehaz)"],
        "trigger": ["nix \\(ceehaz\\), dog knight rpg"],
    },
    "lust_(yamikadesu)": {
        "character": ["lust_(yamikadesu)"],
        "trigger": ["lust \\(yamikadesu\\), mythology"],
    },
    "amelia_raevert": {
        "character": ["amelia_raevert"],
        "trigger": ["amelia raevert, minecraft"],
    },
    "paivio_selanne": {
        "character": ["paivio_selanne"],
        "trigger": ["paivio selanne, rift seekers saga"],
    },
    "mr.hakkai": {"character": ["mr.hakkai"], "trigger": ["mr.hakkai, mythology"]},
    "zurianima_(yagdrassyl)": {
        "character": ["zurianima_(yagdrassyl)"],
        "trigger": ["zurianima \\(yagdrassyl\\), mythology"],
    },
    "the_suit_(ponporio)": {
        "character": ["the_suit_(ponporio)"],
        "trigger": ["the suit \\(ponporio\\), broly culo"],
    },
    "fanny_mcphee": {
        "character": ["fanny_mcphee"],
        "trigger": ["fanny mcphee, grannybase"],
    },
    "zhu_(character)": {
        "character": ["zhu_(character)"],
        "trigger": ["zhu \\(character\\), christmas"],
    },
    "orio_(nonbinary_bunny)": {
        "character": ["orio_(nonbinary_bunny)"],
        "trigger": ["orio \\(nonbinary bunny\\), pokemon"],
    },
    "aster_(nu:_carnival)": {
        "character": ["aster_(nu:_carnival)"],
        "trigger": ["aster \\(nu: carnival\\), nu: carnival"],
    },
    "lakaiger": {"character": ["lakaiger"], "trigger": ["lakaiger, warcraft"]},
    "melvin_(hoppscotch)": {
        "character": ["melvin_(hoppscotch)"],
        "trigger": ["melvin \\(hoppscotch\\), hoppscotch"],
    },
    "xbuimonsama": {"character": ["xbuimonsama"], "trigger": ["xbuimonsama, digimon"]},
    "dj_mixer_(character)": {
        "character": ["dj_mixer_(character)"],
        "trigger": ["dj mixer \\(character\\), teenage mutant ninja turtles"],
    },
    "clover_(totally_spies!)": {
        "character": ["clover_(totally_spies!)"],
        "trigger": ["clover \\(totally spies!\\), totally spies!"],
    },
    "chaosie": {"character": ["chaosie"], "trigger": ["chaosie, mythology"]},
    "tora": {"character": ["tora"], "trigger": ["tora, mythology"]},
    "elvira_(mistress_of_the_dark)": {
        "character": ["elvira_(mistress_of_the_dark)"],
        "trigger": ["elvira \\(mistress of the dark\\), elvira: mistress of the dark"],
    },
    "tsuki": {"character": ["tsuki"], "trigger": ["tsuki, warcraft"]},
    "mandarax": {"character": ["mandarax"], "trigger": ["mandarax, mythology"]},
    "masamune": {"character": ["masamune"], "trigger": ["masamune, pokemon"]},
    "starfire": {"character": ["starfire"], "trigger": ["starfire, dc comics"]},
    "blue_(pokemon)": {
        "character": ["blue_(pokemon)"],
        "trigger": ["blue \\(pokemon\\), pokemon"],
    },
    "noah_(project_geeker)": {
        "character": ["noah_(project_geeker)"],
        "trigger": ["noah \\(project geeker\\), project geeker"],
    },
    "foxy_roxy": {
        "character": ["foxy_roxy"],
        "trigger": ["foxy roxy, brutal paws of fury"],
    },
    "boss_(hamtaro)": {
        "character": ["boss_(hamtaro)"],
        "trigger": ["boss \\(hamtaro\\), hamtaro \\(series\\)"],
    },
    "kanic": {"character": ["kanic"], "trigger": ["kanic, mythology"]},
    "croc": {"character": ["croc"], "trigger": ["croc, croc: legend of the gobbos"]},
    "stefan_(hextra)": {
        "character": ["stefan_(hextra)"],
        "trigger": ["stefan \\(hextra\\), hextra"],
    },
    "dawn_(ymbk)": {
        "character": ["dawn_(ymbk)"],
        "trigger": ["dawn \\(ymbk\\), pokemon"],
    },
    "delilah_(trias)": {
        "character": ["delilah_(trias)"],
        "trigger": ["delilah \\(trias\\), dinosaurs inc."],
    },
    "nadeena": {"character": ["nadeena"], "trigger": ["nadeena, mythology"]},
    "kitara_cydonis": {
        "character": ["kitara_cydonis"],
        "trigger": ["kitara cydonis, mythology"],
    },
    "oola": {"character": ["oola"], "trigger": ["oola, star wars"]},
    "ratharn": {"character": ["ratharn"], "trigger": ["ratharn, mythology"]},
    "peewee": {"character": ["peewee"], "trigger": ["peewee, riot games"]},
    "hound_(character)": {
        "character": ["hound_(character)"],
        "trigger": ["hound \\(character\\), pokemon"],
    },
    "great_grey_wolf_sif": {
        "character": ["great_grey_wolf_sif"],
        "trigger": ["great grey wolf sif, fromsoftware"],
    },
    "xhyra": {"character": ["xhyra"], "trigger": ["xhyra, mythology"]},
    "tamamo": {"character": ["tamamo"], "trigger": ["tamamo, monster girl quest"]},
    "ms._pennypacker": {
        "character": ["ms._pennypacker"],
        "trigger": ["ms. pennypacker, disney"],
    },
    "roscoe_(animal_crossing)": {
        "character": ["roscoe_(animal_crossing)"],
        "trigger": ["roscoe \\(animal crossing\\), animal crossing"],
    },
    "markshark": {"character": ["markshark"], "trigger": ["markshark, mythology"]},
    "barry_torres": {"character": ["barry_torres"], "trigger": ["barry torres"]},
    "stegz": {"character": ["stegz"], "trigger": ["stegz, extreme dinosaurs"]},
    "rainbow_blaze_(mlp)": {
        "character": ["rainbow_blaze_(mlp)"],
        "trigger": ["rainbow blaze \\(mlp\\), my little pony"],
    },
    "diesel_(ralarare)": {
        "character": ["diesel_(ralarare)"],
        "trigger": ["diesel \\(ralarare\\), sonicfox"],
    },
    "atlas_(jelomaus)": {
        "character": ["atlas_(jelomaus)"],
        "trigger": ["atlas \\(jelomaus\\), my little pony"],
    },
    "beta_tyson": {"character": ["beta_tyson"], "trigger": ["beta tyson, mythology"]},
    "lucidum": {"character": ["lucidum"], "trigger": ["lucidum, mythology"]},
    "nate_(dragoneill)": {
        "character": ["nate_(dragoneill)"],
        "trigger": ["nate \\(dragoneill\\), snapchat"],
    },
    "bayron_(character)": {
        "character": ["bayron_(character)"],
        "trigger": ["bayron \\(character\\), nintendo"],
    },
    "kasia_mikolajczyk": {
        "character": ["kasia_mikolajczyk"],
        "trigger": ["kasia mikolajczyk, pokemon"],
    },
    "haze_the_giraking": {
        "character": ["haze_the_giraking"],
        "trigger": ["haze the giraking, pokemon"],
    },
    "magenta_(magenta7)": {
        "character": ["magenta_(magenta7)"],
        "trigger": ["magenta \\(magenta7\\), mythology"],
    },
    "alex_the_rubikang": {
        "character": ["alex_the_rubikang"],
        "trigger": ["alex the rubikang, nintendo"],
    },
    "rika_(desbjust)": {
        "character": ["rika_(desbjust)"],
        "trigger": ["rika \\(desbjust\\), mythology"],
    },
    "bob_(undertale)": {
        "character": ["bob_(undertale)"],
        "trigger": ["bob \\(undertale\\), undertale \\(series\\)"],
    },
    "lucy_(aikega)": {
        "character": ["lucy_(aikega)"],
        "trigger": ["lucy \\(aikega\\), pokemon"],
    },
    "zofie_(fluff-kevlar)": {
        "character": ["zofie_(fluff-kevlar)"],
        "trigger": ["zofie \\(fluff-kevlar\\), halloween"],
    },
    "shurya": {"character": ["shurya"], "trigger": ["shurya, mythology"]},
    "martin_ballamore": {
        "character": ["martin_ballamore"],
        "trigger": ["martin ballamore, square enix"],
    },
    "keiko_tachibana": {
        "character": ["keiko_tachibana"],
        "trigger": ["keiko tachibana, east asian mythology"],
    },
    "corablue_(character)": {
        "character": ["corablue_(character)"],
        "trigger": ["corablue \\(character\\), mythology"],
    },
    "melody_(aseethe)": {
        "character": ["melody_(aseethe)"],
        "trigger": ["melody \\(aseethe\\), bloodborne"],
    },
    "sara_(sailoranna)": {
        "character": ["sara_(sailoranna)"],
        "trigger": ["sara \\(sailoranna\\), mythology"],
    },
    "fuo_(hanadaiteol)": {
        "character": ["fuo_(hanadaiteol)"],
        "trigger": ["fuo \\(hanadaiteol\\), mythology"],
    },
    "rorian_blackrose": {
        "character": ["rorian_blackrose"],
        "trigger": ["rorian blackrose, mythology"],
    },
    "hilda_(warcraft)": {
        "character": ["hilda_(warcraft)"],
        "trigger": ["hilda \\(warcraft\\), warcraft"],
    },
    "yunobo": {"character": ["yunobo"], "trigger": ["yunobo, the legend of zelda"]},
    "dantera_lina": {
        "character": ["dantera_lina"],
        "trigger": ["dantera lina, mythology"],
    },
    "ruth_(sharkstuff)": {
        "character": ["ruth_(sharkstuff)"],
        "trigger": ["ruth \\(sharkstuff\\), source filmmaker"],
    },
    "red_knight_(sirphilliam)": {
        "character": ["red_knight_(sirphilliam)"],
        "trigger": ["red knight \\(sirphilliam\\), mythology"],
    },
    "chelsea_(pearlhead)": {
        "character": ["chelsea_(pearlhead)"],
        "trigger": ["chelsea \\(pearlhead\\), halloween"],
    },
    "king_dice": {
        "character": ["king_dice"],
        "trigger": ["king dice, cuphead \\(game\\)"],
    },
    "lucas_arynn": {"character": ["lucas_arynn"], "trigger": ["lucas arynn, pokemon"]},
    "ian_(braeburned)": {
        "character": ["ian_(braeburned)"],
        "trigger": ["ian \\(braeburned\\), nintendo"],
    },
    "skully_(skully)": {
        "character": ["skully_(skully)"],
        "trigger": ["skully \\(skully\\), mythology"],
    },
    "deadman_joe_velasquez": {
        "character": ["deadman_joe_velasquez"],
        "trigger": ["deadman joe velasquez, mythology"],
    },
    "adrian_vitalis": {
        "character": ["adrian_vitalis"],
        "trigger": ["adrian vitalis, patreon"],
    },
    "daxx_(shirteater18)": {
        "character": ["daxx_(shirteater18)"],
        "trigger": ["daxx \\(shirteater18\\), patreon"],
    },
    "mikomon": {"character": ["mikomon"], "trigger": ["mikomon, digimon"]},
    "brooke_(simplifypm)": {
        "character": ["brooke_(simplifypm)"],
        "trigger": ["brooke \\(simplifypm\\), mythology"],
    },
    "lidigeneer_(lidigeneer)": {
        "character": ["lidigeneer_(lidigeneer)"],
        "trigger": ["lidigeneer \\(lidigeneer\\), mythology"],
    },
    "warfare_freya": {
        "character": ["warfare_freya"],
        "trigger": ["warfare freya, square enix"],
    },
    "wes_(ultilix)": {
        "character": ["wes_(ultilix)"],
        "trigger": ["wes \\(ultilix\\), mythology"],
    },
    "dr._jennifer_dogna": {
        "character": ["dr._jennifer_dogna"],
        "trigger": ["dr. jennifer dogna, pixile studios"],
    },
    "aidan_(doubledog)": {
        "character": ["aidan_(doubledog)"],
        "trigger": ["aidan \\(doubledog\\), mythology"],
    },
    "nisha_(bluedingo)": {
        "character": ["nisha_(bluedingo)"],
        "trigger": ["nisha \\(bluedingo\\), blender \\(software\\)"],
    },
    "eugene_(mao_mao:_heroes_of_pure_heart)": {
        "character": ["eugene_(mao_mao:_heroes_of_pure_heart)"],
        "trigger": ["eugene \\(mao mao: heroes of pure heart\\), cartoon network"],
    },
    "crown_prince_(gunfire_reborn)": {
        "character": ["crown_prince_(gunfire_reborn)"],
        "trigger": ["crown prince \\(gunfire reborn\\), gunfire reborn"],
    },
    "guildmaster_(icma)": {
        "character": ["guildmaster_(icma)"],
        "trigger": ["guildmaster \\(icma\\), pmd: icma"],
    },
    "lya_(scalesindark)": {
        "character": ["lya_(scalesindark)"],
        "trigger": ["lya \\(scalesindark\\), mythology"],
    },
    "sparkychu": {"character": ["sparkychu"], "trigger": ["sparkychu, pokemon"]},
    "sophia_hellstrand": {
        "character": ["sophia_hellstrand"],
        "trigger": ["sophia hellstrand, halloween"],
    },
    "samael_wurlitz": {
        "character": ["samael_wurlitz"],
        "trigger": ["samael wurlitz, rvb revolution"],
    },
    "king_andrias_leviathan": {
        "character": ["king_andrias_leviathan"],
        "trigger": ["king andrias leviathan, disney"],
    },
    "filbo_fiddlepie": {
        "character": ["filbo_fiddlepie"],
        "trigger": ["filbo fiddlepie, bugsnax"],
    },
    "thokk_(invincible)": {
        "character": ["thokk_(invincible)"],
        "trigger": ["thokk \\(invincible\\), image comics"],
    },
    "xiao_hui_(ffjjfjci)": {
        "character": ["xiao_hui_(ffjjfjci)"],
        "trigger": ["xiao hui \\(ffjjfjci\\), pocky"],
    },
    "tobi_(squishy)": {
        "character": ["tobi_(squishy)"],
        "trigger": ["tobi \\(squishy\\), nasa"],
    },
    "arin_(daxhush)": {
        "character": ["arin_(daxhush)"],
        "trigger": ["arin \\(daxhush\\), mythology"],
    },
    "warfare_sally": {
        "character": ["warfare_sally"],
        "trigger": ["warfare sally, sonic the hedgehog \\(series\\)"],
    },
    "kubo_(eebahdeebah)": {
        "character": ["kubo_(eebahdeebah)"],
        "trigger": ["kubo \\(eebahdeebah\\), cartoon network"],
    },
    "dan_(fuf)": {"character": ["dan_(fuf)"], "trigger": ["dan \\(fuf\\), pokemon"]},
    "zeno_(komenuka_inaho)": {
        "character": ["zeno_(komenuka_inaho)"],
        "trigger": ["zeno \\(komenuka inaho\\), mythology"],
    },
    "adam_summers_(forestdale)": {
        "character": ["adam_summers_(forestdale)"],
        "trigger": ["adam summers \\(forestdale\\), forestdale"],
    },
    "reevah_(nuree_art)": {
        "character": ["reevah_(nuree_art)"],
        "trigger": ["reevah \\(nuree art\\), mythology"],
    },
    "kif_yppreah": {"character": ["kif_yppreah"], "trigger": ["kif yppreah"]},
    "shadowheart_(baldur's_gate)": {
        "character": ["shadowheart_(baldur's_gate)"],
        "trigger": ["shadowheart \\(baldur's gate\\), bioware"],
    },
    "bobby_bearhug": {
        "character": ["bobby_bearhug"],
        "trigger": ["bobby bearhug, poppy playtime"],
    },
    "glitter_(kadath)": {
        "character": ["glitter_(kadath)"],
        "trigger": ["glitter \\(kadath\\), patreon"],
    },
    "pathia": {"character": ["pathia"], "trigger": ["pathia, mythology"]},
    "alex_(totally_spies!)": {
        "character": ["alex_(totally_spies!)"],
        "trigger": ["alex \\(totally spies!\\), totally spies!"],
    },
    "zandria": {"character": ["zandria"], "trigger": ["zandria, quest for fun"]},
    "bellecandie": {
        "character": ["bellecandie"],
        "trigger": ["bellecandie, mythology"],
    },
    "bo_(slipco)": {
        "character": ["bo_(slipco)"],
        "trigger": ["bo \\(slipco\\), mythology"],
    },
    "johnny_bravo": {
        "character": ["johnny_bravo"],
        "trigger": ["johnny bravo, cartoon network"],
    },
    "porky_pig": {"character": ["porky_pig"], "trigger": ["porky pig, looney tunes"]},
    "miss_sunflower": {
        "character": ["miss_sunflower"],
        "trigger": ["miss sunflower, conker's bad fur day"],
    },
    "blitzwolfer": {
        "character": ["blitzwolfer"],
        "trigger": ["blitzwolfer, cartoon network"],
    },
    "timmy_turner": {
        "character": ["timmy_turner"],
        "trigger": ["timmy turner, the fairly oddparents"],
    },
    "utsuho_reiuji": {
        "character": ["utsuho_reiuji"],
        "trigger": ["utsuho reiuji, touhou"],
    },
    "shyama": {"character": ["shyama"], "trigger": ["shyama, pokemon"]},
    "stormfront": {
        "character": ["stormfront"],
        "trigger": ["stormfront, my little pony"],
    },
    "zipper_(cdrr)": {
        "character": ["zipper_(cdrr)"],
        "trigger": ["zipper \\(cdrr\\), disney"],
    },
    "ice_king": {"character": ["ice_king"], "trigger": ["ice king, cartoon network"]},
    "lime_ade": {"character": ["lime_ade"], "trigger": ["lime ade, nintendo"]},
    "carnage_(marvel)": {
        "character": ["carnage_(marvel)"],
        "trigger": ["carnage \\(marvel\\), marvel"],
    },
    "serge": {"character": ["serge"], "trigger": ["serge, mythology"]},
    "james_(the-jackal)": {
        "character": ["james_(the-jackal)"],
        "trigger": ["james \\(the-jackal\\), my life with fel"],
    },
    "eddie_(dessy)": {
        "character": ["eddie_(dessy)"],
        "trigger": ["eddie \\(dessy\\), mythology"],
    },
    "roxy_bradingham": {
        "character": ["roxy_bradingham"],
        "trigger": ["roxy bradingham, nintendo"],
    },
    "yukikaze_panettone": {
        "character": ["yukikaze_panettone"],
        "trigger": ["yukikaze panettone, dog days"],
    },
    "baul": {"character": ["baul"], "trigger": ["baul"]},
    "jenny_(powhatan)": {
        "character": ["jenny_(powhatan)"],
        "trigger": ["jenny \\(powhatan\\), powhatan"],
    },
    "bark_the_polar_bear": {
        "character": ["bark_the_polar_bear"],
        "trigger": ["bark the polar bear, sonic the hedgehog \\(series\\)"],
    },
    "ace_the_bat-hound": {
        "character": ["ace_the_bat-hound"],
        "trigger": ["ace the bat-hound, dc comics"],
    },
    "jayfeather_(warriors)": {
        "character": ["jayfeather_(warriors)"],
        "trigger": ["jayfeather \\(warriors\\), warriors \\(book series\\)"],
    },
    "rylai_the_crystal_maiden": {
        "character": ["rylai_the_crystal_maiden"],
        "trigger": ["rylai the crystal maiden, dota"],
    },
    "jabberjaw_(character)": {
        "character": ["jabberjaw_(character)"],
        "trigger": ["jabberjaw \\(character\\), jabberjaw"],
    },
    "daisuke_yomo": {"character": ["daisuke_yomo"], "trigger": ["daisuke yomo"]},
    "polymorph": {"character": ["polymorph"], "trigger": ["polymorph, mythology"]},
    "long_(zerofox)": {
        "character": ["long_(zerofox)"],
        "trigger": ["long \\(zerofox\\), mythology"],
    },
    "angry_kitty": {
        "character": ["angry_kitty"],
        "trigger": ["angry kitty, the lego movie"],
    },
    "ggv": {"character": ["ggv"], "trigger": ["ggv, mythology"]},
    "scarlett_vithica": {
        "character": ["scarlett_vithica"],
        "trigger": ["scarlett vithica, wizards of the coast"],
    },
    "robo-fortune": {
        "character": ["robo-fortune"],
        "trigger": ["robo-fortune, skullgirls"],
    },
    "cirrus_sky": {"character": ["cirrus_sky"], "trigger": ["cirrus sky, mythology"]},
    "shadow_freddy_(fnaf)": {
        "character": ["shadow_freddy_(fnaf)"],
        "trigger": ["shadow freddy \\(fnaf\\), scottgames"],
    },
    "sapphire_lareme_(battler)": {
        "character": ["sapphire_lareme_(battler)"],
        "trigger": ["sapphire lareme \\(battler\\), cub con"],
    },
    "purrl_(animal_crossing)": {
        "character": ["purrl_(animal_crossing)"],
        "trigger": ["purrl \\(animal crossing\\), animal crossing"],
    },
    "marflebark": {"character": ["marflebark"], "trigger": ["marflebark, mythology"]},
    "lottie_(animal_crossing)": {
        "character": ["lottie_(animal_crossing)"],
        "trigger": ["lottie \\(animal crossing\\), animal crossing"],
    },
    "kristoph_wulphenstein": {
        "character": ["kristoph_wulphenstein"],
        "trigger": ["kristoph wulphenstein, meme clothing"],
    },
    "nico_(foxmusk)": {
        "character": ["nico_(foxmusk)"],
        "trigger": ["nico \\(foxmusk\\), mythology"],
    },
    "pinku_(miscon)": {
        "character": ["pinku_(miscon)"],
        "trigger": ["pinku \\(miscon\\), kamen rider"],
    },
    "bee_the_cat": {
        "character": ["bee_the_cat"],
        "trigger": ["bee the cat, mythology"],
    },
    "hanzo_(overwatch)": {
        "character": ["hanzo_(overwatch)"],
        "trigger": ["hanzo \\(overwatch\\), overwatch"],
    },
    "professor_kukui": {
        "character": ["professor_kukui"],
        "trigger": ["professor kukui, pokemon"],
    },
    "rothar": {"character": ["rothar"], "trigger": ["rothar, mythology"]},
    "apollo_caelum": {
        "character": ["apollo_caelum"],
        "trigger": ["apollo caelum, nintendo"],
    },
    "olivia_may": {"character": ["olivia_may"], "trigger": ["olivia may, mythology"]},
    "raptor_matt": {
        "character": ["raptor_matt"],
        "trigger": ["raptor matt, wizards of the coast"],
    },
    "sorlag": {"character": ["sorlag"], "trigger": ["sorlag, quake"]},
    "scarlet_mauve": {
        "character": ["scarlet_mauve"],
        "trigger": ["scarlet mauve, pokemon"],
    },
    "maddy_(bonk)": {"character": ["maddy_(bonk)"], "trigger": ["maddy \\(bonk\\)"]},
    "mimi_(playkids)": {
        "character": ["mimi_(playkids)"],
        "trigger": ["mimi \\(playkids\\), playkids"],
    },
    "lilina": {"character": ["lilina"], "trigger": ["lilina, mythology"]},
    "zeke_the_zorua": {
        "character": ["zeke_the_zorua"],
        "trigger": ["zeke the zorua, pokemon"],
    },
    "leonardo_(rottmnt)": {
        "character": ["leonardo_(rottmnt)"],
        "trigger": ["leonardo \\(rottmnt\\), rise of the teenage mutant ninja turtles"],
    },
    "electrycpynk_(character)": {
        "character": ["electrycpynk_(character)"],
        "trigger": ["electrycpynk \\(character\\), mythology"],
    },
    "cassidy_(ruth66)": {
        "character": ["cassidy_(ruth66)"],
        "trigger": ["cassidy \\(ruth66\\), halloween"],
    },
    "shen_(zummeng)": {
        "character": ["shen_(zummeng)"],
        "trigger": ["shen \\(zummeng\\), patreon"],
    },
    "rodney_(matchaghost)": {
        "character": ["rodney_(matchaghost)"],
        "trigger": ["rodney \\(matchaghost\\), pokemon"],
    },
    "trainer_aliyah": {
        "character": ["trainer_aliyah"],
        "trigger": ["trainer aliyah, pokemon"],
    },
    "nemo_(simplifypm)": {
        "character": ["nemo_(simplifypm)"],
        "trigger": ["nemo \\(simplifypm\\), mythology"],
    },
    "iron_aegis": {
        "character": ["iron_aegis"],
        "trigger": ["iron aegis, my little pony"],
    },
    "yuki_(yukitallorean)": {
        "character": ["yuki_(yukitallorean)"],
        "trigger": ["yuki \\(yukitallorean\\), mythology"],
    },
    "ayrrenth": {"character": ["ayrrenth"], "trigger": ["ayrrenth, mythology"]},
    "shi_yu_(lunarspy)": {
        "character": ["shi_yu_(lunarspy)"],
        "trigger": ["shi yu \\(lunarspy\\), pokemon"],
    },
    "guinea_(interspecies_reviewers)": {
        "character": ["guinea_(interspecies_reviewers)"],
        "trigger": ["guinea \\(interspecies reviewers\\), interspecies reviewers"],
    },
    "marcus_(azathura)": {
        "character": ["marcus_(azathura)"],
        "trigger": ["marcus \\(azathura\\), mythology"],
    },
    "female_operator": {
        "character": ["female_operator"],
        "trigger": ["female operator, lifewonders"],
    },
    "alrik_yeenobotham": {
        "character": ["alrik_yeenobotham"],
        "trigger": ["alrik yeenobotham, san tejon high"],
    },
    "leto_(grimart)": {
        "character": ["leto_(grimart)"],
        "trigger": ["leto \\(grimart\\), ipod"],
    },
    "himbo_stitch": {
        "character": ["himbo_stitch"],
        "trigger": ["himbo stitch, disney"],
    },
    "ye_xiu_(tka)": {
        "character": ["ye_xiu_(tka)"],
        "trigger": ["ye xiu \\(tka\\), the king's avatar"],
    },
    "kapri_(kapri)": {
        "character": ["kapri_(kapri)"],
        "trigger": ["kapri \\(kapri\\), mythology"],
    },
    "tinker_belle": {
        "character": ["tinker_belle"],
        "trigger": ["tinker belle, animal crossing"],
    },
    "juna_june_(lizzyglizzy)": {
        "character": ["juna_june_(lizzyglizzy)"],
        "trigger": ["juna june \\(lizzyglizzy\\), mythology"],
    },
    "gordiethecollie": {
        "character": ["gordiethecollie"],
        "trigger": ["gordiethecollie, jif peanut butter"],
    },
    "hana_(jishinu)": {
        "character": ["hana_(jishinu)"],
        "trigger": ["hana \\(jishinu\\), christmas"],
    },
    "mik_(lonnyk)": {
        "character": ["mik_(lonnyk)"],
        "trigger": ["mik \\(lonnyk\\), no nut november"],
    },
    "sadiend": {"character": ["sadiend"], "trigger": ["sadiend, sports illustrated"]},
    "neth_(aspen)": {
        "character": ["neth_(aspen)"],
        "trigger": ["neth \\(aspen\\), a hat in time"],
    },
    "red_(vixen_logic)": {
        "character": ["red_(vixen_logic)"],
        "trigger": ["red \\(vixen logic\\), vixen logic"],
    },
    "lady_brushfire_(kitfox-crimson)": {
        "character": ["lady_brushfire_(kitfox-crimson)"],
        "trigger": ["lady brushfire \\(kitfox-crimson\\), in our shadow"],
    },
    "benjamin_mcknight": {
        "character": ["benjamin_mcknight"],
        "trigger": ["benjamin mcknight, cavemanon studios"],
    },
    "chieftain_scavenger": {
        "character": ["chieftain_scavenger"],
        "trigger": ["chieftain scavenger, videocult"],
    },
    "terebi": {"character": ["terebi"], "trigger": ["terebi, minecraft"]},
    "sunspot_(fortnite)": {
        "character": ["sunspot_(fortnite)"],
        "trigger": ["sunspot \\(fortnite\\), fortnite"],
    },
    "elma_(tenchi_muyo)": {
        "character": ["elma_(tenchi_muyo)"],
        "trigger": ["elma \\(tenchi muyo\\), tenchi muyo"],
    },
    "flounce_(jay_naylor)": {
        "character": ["flounce_(jay_naylor)"],
        "trigger": ["flounce \\(jay naylor\\), patreon"],
    },
    "bridget_(guilty_gear)": {
        "character": ["bridget_(guilty_gear)"],
        "trigger": ["bridget \\(guilty gear\\), arc system works"],
    },
    "sister_of_battle": {
        "character": ["sister_of_battle"],
        "trigger": ["sister of battle, warhammer \\(franchise\\)"],
    },
    "irma_langinstein": {
        "character": ["irma_langinstein"],
        "trigger": ["irma langinstein, teenage mutant ninja turtles"],
    },
    "itreyu": {"character": ["itreyu"], "trigger": ["itreyu, palm of my hand"]},
    "nina_morena_agil": {
        "character": ["nina_morena_agil"],
        "trigger": ["nina morena agil, good cheese"],
    },
    "shub-niggurath_(h.p._lovecraft)": {
        "character": ["shub-niggurath_(h.p._lovecraft)"],
        "trigger": ["shub-niggurath \\(h.p. lovecraft\\), cthulhu mythos"],
    },
    "slither_(slither)": {
        "character": ["slither_(slither)"],
        "trigger": ["slither \\(slither\\), mythology"],
    },
    "flir_(rabbit)": {
        "character": ["flir_(rabbit)"],
        "trigger": ["flir \\(rabbit\\), cubi \\(characters\\)"],
    },
    "iratu_elexion": {
        "character": ["iratu_elexion"],
        "trigger": ["iratu elexion, slightly damned"],
    },
    "vulpa_(mariano)": {
        "character": ["vulpa_(mariano)"],
        "trigger": ["vulpa \\(mariano\\), square enix"],
    },
    "sir_aaron": {"character": ["sir_aaron"], "trigger": ["sir aaron, pokemon"]},
    "zangief": {"character": ["zangief"], "trigger": ["zangief, capcom"]},
    "thorphax": {"character": ["thorphax"], "trigger": ["thorphax, mythology"]},
    "swiftayama": {"character": ["swiftayama"], "trigger": ["swiftayama, nintendo"]},
    "gerrark": {"character": ["gerrark"], "trigger": ["gerrark, the jungle book"]},
    "ikobi": {"character": ["ikobi"], "trigger": ["ikobi, mythology"]},
    "impa": {"character": ["impa"], "trigger": ["impa, the legend of zelda"]},
    "lassie_lunaris": {
        "character": ["lassie_lunaris"],
        "trigger": ["lassie lunaris, mythology"],
    },
    "rocket_j._squirrel": {
        "character": ["rocket_j._squirrel"],
        "trigger": ["rocket j. squirrel, jay ward productions"],
    },
    "amelia_cresci": {
        "character": ["amelia_cresci"],
        "trigger": ["amelia cresci, mythology"],
    },
    "topaz_(sonic)": {
        "character": ["topaz_(sonic)"],
        "trigger": ["topaz \\(sonic\\), sonic the hedgehog \\(series\\)"],
    },
    "samantha_arrow": {
        "character": ["samantha_arrow"],
        "trigger": ["samantha arrow, 4th of july"],
    },
    "helen_dish": {
        "character": ["helen_dish"],
        "trigger": ["helen dish, sabrina online"],
    },
    "mrs._felicity_fox": {
        "character": ["mrs._felicity_fox"],
        "trigger": ["mrs. felicity fox, fantastic mr. fox"],
    },
    "john_constantine": {
        "character": ["john_constantine"],
        "trigger": ["john constantine, dc comics"],
    },
    "alphina": {"character": ["alphina"], "trigger": ["alphina, mythology"]},
    "independence": {
        "character": ["independence"],
        "trigger": ["independence, h.w.t. studios"],
    },
    "aj_(ajisthebest)": {
        "character": ["aj_(ajisthebest)"],
        "trigger": ["aj \\(ajisthebest\\), nintendo"],
    },
    "ysera": {"character": ["ysera"], "trigger": ["ysera, warcraft"]},
    "lifty_(htf)": {
        "character": ["lifty_(htf)"],
        "trigger": ["lifty \\(htf\\), happy tree friends"],
    },
    "skyican": {"character": ["skyican"], "trigger": ["skyican, mythology"]},
    "merengue_(animal_crossing)": {
        "character": ["merengue_(animal_crossing)"],
        "trigger": ["merengue \\(animal crossing\\), animal crossing"],
    },
    "ryken": {"character": ["ryken"], "trigger": ["ryken, mythology"]},
    "schmozy": {"character": ["schmozy"], "trigger": ["schmozy, mythology"]},
    "pitu_le_pew": {
        "character": ["pitu_le_pew"],
        "trigger": ["pitu le pew, warner brothers"],
    },
    "darkeye": {"character": ["darkeye"], "trigger": ["darkeye, nintendo"]},
    "horace_horsecollar": {
        "character": ["horace_horsecollar"],
        "trigger": ["horace horsecollar, disney"],
    },
    "omega-xis": {"character": ["omega-xis"], "trigger": ["omega-xis, capcom"]},
    "kohaku_(fuu)": {
        "character": ["kohaku_(fuu)"],
        "trigger": ["kohaku \\(fuu\\), furry scale"],
    },
    "eve_(sloss)": {
        "character": ["eve_(sloss)"],
        "trigger": ["eve \\(sloss\\), christmas"],
    },
    "ceylon_(stitchy626)": {
        "character": ["ceylon_(stitchy626)"],
        "trigger": ["ceylon \\(stitchy626\\), mythology"],
    },
    "ravebounce": {
        "character": ["ravebounce"],
        "trigger": ["ravebounce, my little pony"],
    },
    "aven-fawn": {"character": ["aven-fawn"], "trigger": ["aven-fawn, netflix"]},
    "katsuma": {"character": ["katsuma"], "trigger": ["katsuma, moshi monsters"]},
    "tarrin": {"character": ["tarrin"], "trigger": ["tarrin, mythology"]},
    "tteyuu": {"character": ["tteyuu"], "trigger": ["tteyuu, youtube"]},
    "dractaco": {"character": ["dractaco"], "trigger": ["dractaco, mythology"]},
    "aura_(aurastrasza)": {
        "character": ["aura_(aurastrasza)"],
        "trigger": ["aura \\(aurastrasza\\), warcraft"],
    },
    "eugene_(raichupuppy)": {
        "character": ["eugene_(raichupuppy)"],
        "trigger": ["eugene \\(raichupuppy\\), tamagotchi"],
    },
    "karen_plankton": {
        "character": ["karen_plankton"],
        "trigger": ["karen plankton, spongebob squarepants"],
    },
    "harley_(copperback01)": {
        "character": ["harley_(copperback01)"],
        "trigger": ["harley \\(copperback01\\), christmas"],
    },
    "selkie_(fire_emblem_fates)": {
        "character": ["selkie_(fire_emblem_fates)"],
        "trigger": ["selkie \\(fire emblem fates\\), fire emblem fates"],
    },
    "nystemy_(character)": {
        "character": ["nystemy_(character)"],
        "trigger": ["nystemy \\(character\\), mythology"],
    },
    "clair_(seel_kaiser)": {
        "character": ["clair_(seel_kaiser)"],
        "trigger": ["clair \\(seel kaiser\\), youtube"],
    },
    "ichimatsu_matsuno": {
        "character": ["ichimatsu_matsuno"],
        "trigger": ["ichimatsu matsuno, mr. osomatsu"],
    },
    "keaton_(fire_emblem)": {
        "character": ["keaton_(fire_emblem)"],
        "trigger": ["keaton \\(fire emblem\\), fire emblem"],
    },
    "lopin": {"character": ["lopin"], "trigger": ["lopin, out-of-placers"]},
    "lyn_(z-ray)": {
        "character": ["lyn_(z-ray)"],
        "trigger": ["lyn \\(z-ray\\), mythology"],
    },
    "sensh_the_cat": {
        "character": ["sensh_the_cat"],
        "trigger": ["sensh the cat, sonic the hedgehog \\(series\\)"],
    },
    "mash_kyrielight": {
        "character": ["mash_kyrielight"],
        "trigger": ["mash kyrielight, type-moon"],
    },
    "robert_hayes": {
        "character": ["robert_hayes"],
        "trigger": ["robert hayes, apple inc."],
    },
    "rose_(natsunomeryu)": {
        "character": ["rose_(natsunomeryu)"],
        "trigger": ["rose \\(natsunomeryu\\), mythology"],
    },
    "scott_ryder": {
        "character": ["scott_ryder"],
        "trigger": ["scott ryder, mass effect"],
    },
    "maomi_(doomdutch)": {
        "character": ["maomi_(doomdutch)"],
        "trigger": ["maomi \\(doomdutch\\), source filmmaker"],
    },
    "tsunami_(wof)": {
        "character": ["tsunami_(wof)"],
        "trigger": ["tsunami \\(wof\\), mythology"],
    },
    "fulvus": {"character": ["fulvus"], "trigger": ["fulvus, patreon"]},
    "camber": {"character": ["camber"], "trigger": ["camber, my little pony"]},
    "junior_(playkids)": {
        "character": ["junior_(playkids)"],
        "trigger": ["junior \\(playkids\\), playkids"],
    },
    "guilmon_(bacn)": {
        "character": ["guilmon_(bacn)"],
        "trigger": ["guilmon \\(bacn\\), digimon"],
    },
    "sofi_(aygee)": {
        "character": ["sofi_(aygee)"],
        "trigger": ["sofi \\(aygee\\), mythology"],
    },
    "sammy_(buxbi)": {
        "character": ["sammy_(buxbi)"],
        "trigger": ["sammy \\(buxbi\\), dungeons and dragons"],
    },
    "dax_(dax1)": {
        "character": ["dax_(dax1)"],
        "trigger": ["dax \\(dax1\\), jak and daxter"],
    },
    "thunderbird_(tas)": {
        "character": ["thunderbird_(tas)"],
        "trigger": ["thunderbird \\(tas\\), lifewonders"],
    },
    "juniper_(wanderlust)": {
        "character": ["juniper_(wanderlust)"],
        "trigger": ["juniper \\(wanderlust\\), pokemon"],
    },
    "neonatta": {"character": ["neonatta"], "trigger": ["neonatta, tera online"]},
    "ravios": {"character": ["ravios"], "trigger": ["ravios, mythology"]},
    "oliver_(fuel)": {
        "character": ["oliver_(fuel)"],
        "trigger": ["oliver \\(fuel\\), disney"],
    },
    "finvi": {"character": ["finvi"], "trigger": ["finvi, mentos"]},
    "komi_shouko": {
        "character": ["komi_shouko"],
        "trigger": ["komi shouko, komi-san wa komyushou desu"],
    },
    "malkai_(malkaiwot)": {
        "character": ["malkai_(malkaiwot)"],
        "trigger": ["malkai \\(malkaiwot\\), cartoon network"],
    },
    "celes_traydor": {
        "character": ["celes_traydor"],
        "trigger": ["celes traydor, nintendo"],
    },
    "amber_faegal": {
        "character": ["amber_faegal"],
        "trigger": ["amber faegal, caelum sky"],
    },
    "gluttony_(changing_fates)": {
        "character": ["gluttony_(changing_fates)"],
        "trigger": ["gluttony \\(changing fates\\), east asian mythology"],
    },
    "prazite": {"character": ["prazite"], "trigger": ["prazite, pokemon"]},
    "nigel_(zummeng)": {
        "character": ["nigel_(zummeng)"],
        "trigger": ["nigel \\(zummeng\\), supreme"],
    },
    "zel_(interspecies_reviewers)": {
        "character": ["zel_(interspecies_reviewers)"],
        "trigger": ["zel \\(interspecies reviewers\\), interspecies reviewers"],
    },
    "avalee": {"character": ["avalee"], "trigger": ["avalee, monster hunter"]},
    "miss_thompson_(tegerio)": {
        "character": ["miss_thompson_(tegerio)"],
        "trigger": ["miss thompson \\(tegerio\\), zandar's saga"],
    },
    "pepper_(pepperfox)": {
        "character": ["pepper_(pepperfox)"],
        "trigger": ["pepper \\(pepperfox\\), mythology"],
    },
    "steve_(glassshine)": {
        "character": ["steve_(glassshine)"],
        "trigger": ["steve \\(glassshine\\), mythology"],
    },
    "julia_caernarvon": {
        "character": ["julia_caernarvon"],
        "trigger": ["julia caernarvon, pokemon"],
    },
    "emira_blight": {
        "character": ["emira_blight"],
        "trigger": ["emira blight, disney"],
    },
    "snorpington_fizzlebean": {
        "character": ["snorpington_fizzlebean"],
        "trigger": ["snorpington fizzlebean, bugsnax"],
    },
    "mime_(ekkokenight)": {
        "character": ["mime_(ekkokenight)"],
        "trigger": ["mime \\(ekkokenight\\), nintendo"],
    },
    "balder_(phoenix0310)": {
        "character": ["balder_(phoenix0310)"],
        "trigger": ["balder \\(phoenix0310\\), mythology"],
    },
    "sakuroma_(retrospecter)": {
        "character": ["sakuroma_(retrospecter)"],
        "trigger": ["sakuroma \\(retrospecter\\), mythology"],
    },
    "fanta_(carrotfanta)": {
        "character": ["fanta_(carrotfanta)"],
        "trigger": ["fanta \\(carrotfanta\\), nintendo"],
    },
    "goth_izzy_(mlp)": {
        "character": ["goth_izzy_(mlp)"],
        "trigger": ["goth izzy \\(mlp\\), mlp g5"],
    },
    "v_(murder_drones)": {
        "character": ["v_(murder_drones)"],
        "trigger": ["v \\(murder drones\\), murder drones"],
    },
    "marie_(cally3d)": {
        "character": ["marie_(cally3d)"],
        "trigger": ["marie \\(cally3d\\), five nights at freddy's"],
    },
    "grace_(floa)": {
        "character": ["grace_(floa)"],
        "trigger": ["grace \\(floa\\), pokemon"],
    },
    "jae_(comatose)": {
        "character": ["jae_(comatose)"],
        "trigger": ["jae \\(comatose\\), valentine's day"],
    },
    "oskar_(2ndvoice)": {
        "character": ["oskar_(2ndvoice)"],
        "trigger": ["oskar \\(2ndvoice\\), mythology"],
    },
    "lexington_ulfric_izunia": {
        "character": ["lexington_ulfric_izunia"],
        "trigger": ["lexington ulfric izunia, european mythology"],
    },
    "ralan_(thepatchedragon)": {
        "character": ["ralan_(thepatchedragon)"],
        "trigger": ["ralan \\(thepatchedragon\\), mythology"],
    },
    "drew_(dislyte)": {
        "character": ["drew_(dislyte)"],
        "trigger": ["drew \\(dislyte\\), dislyte"],
    },
    "zephyri_q_wolf": {
        "character": ["zephyri_q_wolf"],
        "trigger": ["zephyri q wolf, halloween"],
    },
    "ratboy": {
        "character": ["ratboy"],
        "trigger": ["ratboy, warhammer \\(franchise\\)"],
    },
    "ninja_(sonic_frontiers)": {
        "character": ["ninja_(sonic_frontiers)"],
        "trigger": ["ninja \\(sonic frontiers\\), sonic frontiers"],
    },
    "kasey_kyle": {"character": ["kasey_kyle"], "trigger": ["kasey kyle, nintendo"]},
    "peppino_spaghetti": {
        "character": ["peppino_spaghetti"],
        "trigger": ["peppino spaghetti, pizza tower"],
    },
    "ryu_(randomtanstudio)": {
        "character": ["ryu_(randomtanstudio)"],
        "trigger": ["ryu \\(randomtanstudio\\), mythology"],
    },
    "busty_bunny": {"character": ["busty_bunny"], "trigger": ["busty bunny, easter"]},
    "shiuk_(character)": {
        "character": ["shiuk_(character)"],
        "trigger": ["shiuk \\(character\\), mythology"],
    },
    "aya_shameimaru": {
        "character": ["aya_shameimaru"],
        "trigger": ["aya shameimaru, touhou"],
    },
    "nana_(ice_climber)": {
        "character": ["nana_(ice_climber)"],
        "trigger": ["nana \\(ice climber\\), ice climber"],
    },
    "asphyxia_lemieux": {
        "character": ["asphyxia_lemieux"],
        "trigger": ["asphyxia lemieux, bendy and the ink machine"],
    },
    "casey_jones": {
        "character": ["casey_jones"],
        "trigger": ["casey jones, teenage mutant ninja turtles"],
    },
    "rosechu_(character)": {
        "character": ["rosechu_(character)"],
        "trigger": ["rosechu \\(character\\), sonichu \\(series\\)"],
    },
    "katara": {
        "character": ["katara"],
        "trigger": ["katara, avatar: the last airbender"],
    },
    "dark_dragon_(american_dragon)": {
        "character": ["dark_dragon_(american_dragon)"],
        "trigger": ["dark dragon \\(american dragon\\), disney"],
    },
    "majora": {"character": ["majora"], "trigger": ["majora, nintendo"]},
    "miss_brush_(brushfire)": {
        "character": ["miss_brush_(brushfire)"],
        "trigger": ["miss brush \\(brushfire\\), the stable"],
    },
    "tasmanian_devil_(looney_tunes)": {
        "character": ["tasmanian_devil_(looney_tunes)"],
        "trigger": ["tasmanian devil \\(looney tunes\\), looney tunes"],
    },
    "han_solo": {"character": ["han_solo"], "trigger": ["han solo, star wars"]},
    "joel_calley": {
        "character": ["joel_calley"],
        "trigger": ["joel calley, concession"],
    },
    "krinele_fullin": {
        "character": ["krinele_fullin"],
        "trigger": ["krinele fullin, christmas"],
    },
    "jade_chan": {
        "character": ["jade_chan"],
        "trigger": ["jade chan, jackie chan adventures"],
    },
    "osiris": {"character": ["osiris"], "trigger": ["osiris, mythology"]},
    "swoop_(philadelphia_eagles)": {
        "character": ["swoop_(philadelphia_eagles)"],
        "trigger": ["swoop \\(philadelphia eagles\\), nfl"],
    },
    "terrador": {"character": ["terrador"], "trigger": ["terrador, spyro the dragon"]},
    "creepy_susie": {
        "character": ["creepy_susie"],
        "trigger": ["creepy susie, the oblongs"],
    },
    "javeloz": {"character": ["javeloz"], "trigger": ["javeloz, mythology"]},
    "jem_(hornedproxy)": {
        "character": ["jem_(hornedproxy)"],
        "trigger": ["jem \\(hornedproxy\\), pokemon"],
    },
    "loki_(marvel)": {
        "character": ["loki_(marvel)"],
        "trigger": ["loki \\(marvel\\), marvel"],
    },
    "drad": {"character": ["drad"], "trigger": ["drad, mythology"]},
    "modjo": {"character": ["modjo"], "trigger": ["modjo, mythology"]},
    "tealmarket": {"character": ["tealmarket"], "trigger": ["tealmarket, mythology"]},
    "ame_(wolf_children)": {
        "character": ["ame_(wolf_children)"],
        "trigger": ["ame \\(wolf children\\), wolf children"],
    },
    "julia_(werefox)": {
        "character": ["julia_(werefox)"],
        "trigger": ["julia \\(werefox\\), mythology"],
    },
    "danger_mouse": {
        "character": ["danger_mouse"],
        "trigger": ["danger mouse, danger mouse \\(series\\)"],
    },
    "mek_(harmarist)": {
        "character": ["mek_(harmarist)"],
        "trigger": ["mek \\(harmarist\\), sheath and knife"],
    },
    "snake_(animal_crossing)": {
        "character": ["snake_(animal_crossing)"],
        "trigger": ["snake \\(animal crossing\\), animal crossing"],
    },
    "feyyore": {"character": ["feyyore"], "trigger": ["feyyore, blender cycles"]},
    "eris_(legends_of_chima)": {
        "character": ["eris_(legends_of_chima)"],
        "trigger": ["eris \\(legends of chima\\), legends of chima"],
    },
    "genji_(animal_crossing)": {
        "character": ["genji_(animal_crossing)"],
        "trigger": ["genji \\(animal crossing\\), animal crossing"],
    },
    "calypso_tayro": {
        "character": ["calypso_tayro"],
        "trigger": ["calypso tayro, nirvana"],
    },
    "sevrah": {"character": ["sevrah"], "trigger": ["sevrah"]},
    "chandra_(abluedeer)": {
        "character": ["chandra_(abluedeer)"],
        "trigger": ["chandra \\(abluedeer\\), moon lace"],
    },
    "ty_(zp92)": {"character": ["ty_(zp92)"], "trigger": ["ty \\(zp92\\), mythology"]},
    "mappy_(character)": {
        "character": ["mappy_(character)"],
        "trigger": ["mappy \\(character\\), mappy"],
    },
    "grand_councilwoman": {
        "character": ["grand_councilwoman"],
        "trigger": ["grand councilwoman, disney"],
    },
    "corporal_the_polar_bear": {
        "character": ["corporal_the_polar_bear"],
        "trigger": ["corporal the polar bear, dreamworks"],
    },
    "mason_hamrell": {
        "character": ["mason_hamrell"],
        "trigger": ["mason hamrell, uberquest"],
    },
    "ellie_cooper": {
        "character": ["ellie_cooper"],
        "trigger": ["ellie cooper, warcraft"],
    },
    "connie_savannah": {
        "character": ["connie_savannah"],
        "trigger": ["connie savannah, nintendo"],
    },
    "the_sole_survivor_(fallout)": {
        "character": ["the_sole_survivor_(fallout)"],
        "trigger": ["the sole survivor \\(fallout\\), fallout"],
    },
    "lilith_(jl2154)": {
        "character": ["lilith_(jl2154)"],
        "trigger": ["lilith \\(jl2154\\), mythology"],
    },
    "gallar_(nnecgrau)": {
        "character": ["gallar_(nnecgrau)"],
        "trigger": ["gallar \\(nnecgrau\\), mythology"],
    },
    "shadowthedemon": {
        "character": ["shadowthedemon"],
        "trigger": ["shadowthedemon, mythology"],
    },
    "midnight_sparkle_(eg)": {
        "character": ["midnight_sparkle_(eg)"],
        "trigger": ["midnight sparkle \\(eg\\), my little pony"],
    },
    "aronai": {"character": ["aronai"], "trigger": ["aronai, mythology"]},
    "petrabyte_incast": {
        "character": ["petrabyte_incast"],
        "trigger": ["petrabyte incast, my little pony"],
    },
    "mazzy_techna": {
        "character": ["mazzy_techna"],
        "trigger": ["mazzy techna, mythology"],
    },
    "rod_garth": {"character": ["rod_garth"], "trigger": ["rod garth, dreamkeepers"]},
    "amber_(kabscorner)": {
        "character": ["amber_(kabscorner)"],
        "trigger": ["amber \\(kabscorner\\), patreon"],
    },
    "mara_(scorpdk)": {
        "character": ["mara_(scorpdk)"],
        "trigger": ["mara \\(scorpdk\\), meme clothing"],
    },
    "magda_wakeman": {
        "character": ["magda_wakeman"],
        "trigger": ["magda wakeman, microsoft"],
    },
    "luna_(roflfox)": {
        "character": ["luna_(roflfox)"],
        "trigger": ["luna \\(roflfox\\), pokemon"],
    },
    "radicles": {"character": ["radicles"], "trigger": ["radicles, cartoon network"]},
    "avery_(vir-no-vigoratus)": {
        "character": ["avery_(vir-no-vigoratus)"],
        "trigger": ["avery \\(vir-no-vigoratus\\), halloween"],
    },
    "nic_(nicopossum)": {
        "character": ["nic_(nicopossum)"],
        "trigger": ["nic \\(nicopossum\\), patreon"],
    },
    "felix_joyful": {
        "character": ["felix_joyful"],
        "trigger": ["felix joyful, mythology"],
    },
    "ettie": {"character": ["ettie"], "trigger": ["ettie, fifa"]},
    "zourik_(zourik)": {
        "character": ["zourik_(zourik)"],
        "trigger": ["zourik \\(zourik\\), mythology"],
    },
    "omar_mercado": {
        "character": ["omar_mercado"],
        "trigger": ["omar mercado, patreon"],
    },
    "azalea_(sylmin)": {
        "character": ["azalea_(sylmin)"],
        "trigger": ["azalea \\(sylmin\\), pokemon"],
    },
    "feniks_felstorm": {
        "character": ["feniks_felstorm"],
        "trigger": ["feniks felstorm, warcraft"],
    },
    "flynn_moore": {
        "character": ["flynn_moore"],
        "trigger": ["flynn moore, echo \\(game\\)"],
    },
    "markus_devore": {
        "character": ["markus_devore"],
        "trigger": ["markus devore, pokemon"],
    },
    "maite_(elcondedeleon)": {
        "character": ["maite_(elcondedeleon)"],
        "trigger": ["maite \\(elcondedeleon\\), mythology"],
    },
    "oasis_(character)": {
        "character": ["oasis_(character)"],
        "trigger": ["oasis \\(character\\), taboo tails \\(copyright\\)"],
    },
    "khamira": {"character": ["khamira"], "trigger": ["khamira, bethesda softworks"]},
    "lily_(funkybun)": {
        "character": ["lily_(funkybun)"],
        "trigger": ["lily \\(funkybun\\), halloween"],
    },
    "kay_rox": {"character": ["kay_rox"], "trigger": ["kay rox, mythology"]},
    "gnorr": {"character": ["gnorr"], "trigger": ["gnorr, wizards of the coast"]},
    "aphrodite_the_absol": {
        "character": ["aphrodite_the_absol"],
        "trigger": ["aphrodite the absol, pokemon"],
    },
    "belle_morgan": {
        "character": ["belle_morgan"],
        "trigger": ["belle morgan, re: strained"],
    },
    "bro_wolffox": {
        "character": ["bro_wolffox"],
        "trigger": ["bro wolffox, halloween"],
    },
    "king_tangu": {"character": ["king_tangu"], "trigger": ["king tangu, nintendo"]},
    "sirocco_zephyrine": {
        "character": ["sirocco_zephyrine"],
        "trigger": ["sirocco zephyrine, square enix"],
    },
    "phoebe_(felino)": {
        "character": ["phoebe_(felino)"],
        "trigger": ["phoebe \\(felino\\), nintendo"],
    },
    "milky_(interspecies_reviewers)": {
        "character": ["milky_(interspecies_reviewers)"],
        "trigger": ["milky \\(interspecies reviewers\\), interspecies reviewers"],
    },
    "aurora_(kamikazekit)": {
        "character": ["aurora_(kamikazekit)"],
        "trigger": ["aurora \\(kamikazekit\\), mythology"],
    },
    "missy_(napalm_express)": {
        "character": ["missy_(napalm_express)"],
        "trigger": ["missy \\(napalm express\\), skittles \\(candy\\)"],
    },
    "wendy_(bluey)": {
        "character": ["wendy_(bluey)"],
        "trigger": ["wendy \\(bluey\\), bluey \\(series\\)"],
    },
    "mizumi_(pyrojey)": {
        "character": ["mizumi_(pyrojey)"],
        "trigger": ["mizumi \\(pyrojey\\), pokemon"],
    },
    "percy_lynxoln_(callmewritefag)": {
        "character": ["percy_lynxoln_(callmewritefag)"],
        "trigger": ["percy lynxoln \\(callmewritefag\\), swat kats"],
    },
    "calie_(s2-freak)": {
        "character": ["calie_(s2-freak)"],
        "trigger": ["calie \\(s2-freak\\), mythology"],
    },
    "garbage_(dogs_in_space)": {
        "character": ["garbage_(dogs_in_space)"],
        "trigger": ["garbage \\(dogs in space\\), dogs in space"],
    },
    "gidoniko_(doneru)": {
        "character": ["gidoniko_(doneru)"],
        "trigger": ["gidoniko \\(doneru\\), mythology"],
    },
    "alphi_(nightbirby)": {
        "character": ["alphi_(nightbirby)"],
        "trigger": ["alphi \\(nightbirby\\), monster hunter"],
    },
    "jesam_(jesam)": {
        "character": ["jesam_(jesam)"],
        "trigger": ["jesam \\(jesam\\), halloween"],
    },
    "dolan_(shane_frost)": {
        "character": ["dolan_(shane_frost)"],
        "trigger": ["dolan \\(shane frost\\), mythology"],
    },
    "drakkor": {"character": ["drakkor"], "trigger": ["drakkor, mythology"]},
    "vera_(vera)": {
        "character": ["vera_(vera)"],
        "trigger": ["vera \\(vera\\), mythology"],
    },
    "toon_patrol": {"character": ["toon_patrol"], "trigger": ["toon patrol, disney"]},
    "mnementh": {"character": ["mnementh"], "trigger": ["mnementh, mythology"]},
    "hazel_(the_sword_in_the_stone)": {
        "character": ["hazel_(the_sword_in_the_stone)"],
        "trigger": ["hazel \\(the sword in the stone\\), the sword in the stone"],
    },
    "reynard_the_fox": {
        "character": ["reynard_the_fox"],
        "trigger": ["reynard the fox, public domain"],
    },
    "amalia_sheran_sharm": {
        "character": ["amalia_sheran_sharm"],
        "trigger": ["amalia sheran sharm, ankama"],
    },
    "jinx_(dc)": {"character": ["jinx_(dc)"], "trigger": ["jinx \\(dc\\), dc comics"]},
    "ra": {"character": ["ra"], "trigger": ["ra, egyptian mythology"]},
    "leoian": {"character": ["leoian"], "trigger": ["leoian, project2nd"]},
    "giro": {"character": ["giro"], "trigger": ["giro, mythology"]},
    "roki": {"character": ["roki"], "trigger": ["roki, christmas"]},
    "taratsu_(character)": {
        "character": ["taratsu_(character)"],
        "trigger": ["taratsu \\(character\\), christmas"],
    },
    "synthia_vice": {
        "character": ["synthia_vice"],
        "trigger": ["synthia vice, christmas"],
    },
    "sidern_brethencourt": {
        "character": ["sidern_brethencourt"],
        "trigger": ["sidern brethencourt, mythology"],
    },
    "alexi_tishen": {
        "character": ["alexi_tishen"],
        "trigger": ["alexi tishen, mythology"],
    },
    "wild_fire_(mlp)": {
        "character": ["wild_fire_(mlp)"],
        "trigger": ["wild fire \\(mlp\\), my little pony"],
    },
    "vi_(lol)": {"character": ["vi_(lol)"], "trigger": ["vi \\(lol\\), riot games"]},
    "jake_(blazingpelt)": {
        "character": ["jake_(blazingpelt)"],
        "trigger": ["jake \\(blazingpelt\\), nintendo"],
    },
    "sirus": {"character": ["sirus"], "trigger": ["sirus, morenatsu"]},
    "xarda": {"character": ["xarda"], "trigger": ["xarda, mythology"]},
    "riff_(riff34)": {
        "character": ["riff_(riff34)"],
        "trigger": ["riff \\(riff34\\), mythology"],
    },
    "mistress_mare-velous_(mlp)": {
        "character": ["mistress_mare-velous_(mlp)"],
        "trigger": ["mistress mare-velous \\(mlp\\), my little pony"],
    },
    "ashleigh": {"character": ["ashleigh"], "trigger": ["ashleigh, mythology"]},
    "bessy_(here_there_be_dragons)": {
        "character": ["bessy_(here_there_be_dragons)"],
        "trigger": ["bessy \\(here there be dragons\\), here there be dragons"],
    },
    "penny_(tits)": {
        "character": ["penny_(tits)"],
        "trigger": ["penny \\(tits\\), trials in tainted space"],
    },
    "alivia": {"character": ["alivia"], "trigger": ["alivia, mythology"]},
    "decker": {"character": ["decker"], "trigger": ["decker, go! go! hypergrind"]},
    "flame_(spyro)": {
        "character": ["flame_(spyro)"],
        "trigger": ["flame \\(spyro\\), mythology"],
    },
    "cayes": {"character": ["cayes"], "trigger": ["cayes, mythology"]},
    "gruftine": {
        "character": ["gruftine"],
        "trigger": ["gruftine, school for vampires"],
    },
    "xanderg": {"character": ["xanderg"], "trigger": ["xanderg, mythology"]},
    "kit_(powfooo)": {
        "character": ["kit_(powfooo)"],
        "trigger": ["kit \\(powfooo\\), patreon"],
    },
    "vrock": {"character": ["vrock"], "trigger": ["vrock, tumblr"]},
    "artimus_(character)": {
        "character": ["artimus_(character)"],
        "trigger": ["artimus \\(character\\), mythology"],
    },
    "goatdog": {"character": ["goatdog"], "trigger": ["goatdog, mythology"]},
    "keiren_(twokinds)": {
        "character": ["keiren_(twokinds)"],
        "trigger": ["keiren \\(twokinds\\), twokinds"],
    },
    "egan": {"character": ["egan"], "trigger": ["egan, christmas"]},
    "naomi_(mastergodai)": {
        "character": ["naomi_(mastergodai)"],
        "trigger": ["naomi \\(mastergodai\\), rascals"],
    },
    "khloe_prower": {
        "character": ["khloe_prower"],
        "trigger": ["khloe prower, nintendo"],
    },
    "carmine_(sorimori)": {
        "character": ["carmine_(sorimori)"],
        "trigger": ["carmine \\(sorimori\\), mythology"],
    },
    "frisky_(under(her)tail)": {
        "character": ["frisky_(under(her)tail)"],
        "trigger": ["frisky \\(under(her)tail\\), undertale \\(series\\)"],
    },
    "kaelyn_idow": {
        "character": ["kaelyn_idow"],
        "trigger": ["kaelyn idow, mythology"],
    },
    "nari_oakes": {
        "character": ["nari_oakes"],
        "trigger": ["nari oakes, h.w.t. studios"],
    },
    "pickle-pee": {
        "character": ["pickle-pee"],
        "trigger": ["pickle-pee, fromsoftware"],
    },
    "iotran_(character)": {
        "character": ["iotran_(character)"],
        "trigger": ["iotran \\(character\\), iotran"],
    },
    "bongo_(dad)": {
        "character": ["bongo_(dad)"],
        "trigger": ["bongo \\(dad\\), kyllo and bongo"],
    },
    "sicmop_(character)": {
        "character": ["sicmop_(character)"],
        "trigger": ["sicmop \\(character\\), black metal"],
    },
    "kate_(hioshiru)": {
        "character": ["kate_(hioshiru)"],
        "trigger": ["kate \\(hioshiru\\), my little pony"],
    },
    "halley": {"character": ["halley"], "trigger": ["halley, mythology"]},
    "glacial_(wintrygale)": {
        "character": ["glacial_(wintrygale)"],
        "trigger": ["glacial \\(wintrygale\\), pokemon"],
    },
    "roy_(chuki)": {
        "character": ["roy_(chuki)"],
        "trigger": ["roy \\(chuki\\), nintendo"],
    },
    "melkah": {"character": ["melkah"], "trigger": ["melkah, tumblr"]},
    "bryce_(angels_with_scaly_wings)": {
        "character": ["bryce_(angels_with_scaly_wings)"],
        "trigger": ["bryce \\(angels with scaly wings\\), angels with scaly wings"],
    },
    "marte_(gaturo)": {
        "character": ["marte_(gaturo)"],
        "trigger": ["marte \\(gaturo\\), ghostbusters"],
    },
    "minnow_(lemonynade)": {
        "character": ["minnow_(lemonynade)"],
        "trigger": ["minnow \\(lemonynade\\), mythology"],
    },
    "tiramisu_skunk": {
        "character": ["tiramisu_skunk"],
        "trigger": ["tiramisu skunk, fortnite"],
    },
    "isher": {"character": ["isher"], "trigger": ["isher, out-of-placers"]},
    "sylas_(sylasdoggo)": {
        "character": ["sylas_(sylasdoggo)"],
        "trigger": ["sylas \\(sylasdoggo\\), mythology"],
    },
    "amber_(batartcave)": {
        "character": ["amber_(batartcave)"],
        "trigger": ["amber \\(batartcave\\), christmas"],
    },
    "hooves-art_(oc)": {
        "character": ["hooves-art_(oc)"],
        "trigger": ["hooves-art \\(oc\\), my little pony"],
    },
    "bonfire_(bonfirefox)": {
        "character": ["bonfire_(bonfirefox)"],
        "trigger": ["bonfire \\(bonfirefox\\), mythology"],
    },
    "hondra": {"character": ["hondra"], "trigger": ["hondra, mythology"]},
    "mia_perella": {"character": ["mia_perella"], "trigger": ["mia perella, pokemon"]},
    "javisylveon_(mintyspirit)": {
        "character": ["javisylveon_(mintyspirit)"],
        "trigger": ["javisylveon \\(mintyspirit\\), pokemon"],
    },
    "olga_(jenexian)": {
        "character": ["olga_(jenexian)"],
        "trigger": ["olga \\(jenexian\\), mythology"],
    },
    "jockington_(deltarune)": {
        "character": ["jockington_(deltarune)"],
        "trigger": ["jockington \\(deltarune\\), undertale \\(series\\)"],
    },
    "sami_demarco": {
        "character": ["sami_demarco"],
        "trigger": ["sami demarco, mythology"],
    },
    "deepak_(101_dalmatians)": {
        "character": ["deepak_(101_dalmatians)"],
        "trigger": ["deepak \\(101 dalmatians\\), disney"],
    },
    "yuuichi_michimiya": {
        "character": ["yuuichi_michimiya"],
        "trigger": ["yuuichi michimiya, tennis ace"],
    },
    "king_manu": {"character": ["king_manu"], "trigger": ["king manu, mythology"]},
    "vincent_(litterbox_comics)": {
        "character": ["vincent_(litterbox_comics)"],
        "trigger": ["vincent \\(litterbox comics\\), litterbox comics"],
    },
    "avio_(avioylin)": {
        "character": ["avio_(avioylin)"],
        "trigger": ["avio \\(avioylin\\), disney"],
    },
    "fatehunter": {"character": ["fatehunter"], "trigger": ["fatehunter, mythology"]},
    "colleen_(masterofall)": {
        "character": ["colleen_(masterofall)"],
        "trigger": ["colleen \\(masterofall\\), disney"],
    },
    "bayard_zylos": {
        "character": ["bayard_zylos"],
        "trigger": ["bayard zylos, mythology"],
    },
    "koi-chan": {
        "character": ["koi-chan"],
        "trigger": ["koi-chan, real axolotl hours"],
    },
    "sam_yaeger": {
        "character": ["sam_yaeger"],
        "trigger": ["sam yaeger, studio trigger"],
    },
    "dawn_(darkjester)": {
        "character": ["dawn_(darkjester)"],
        "trigger": ["dawn \\(darkjester\\), mythology"],
    },
    "christopher_(zummeng)": {
        "character": ["christopher_(zummeng)"],
        "trigger": ["christopher \\(zummeng\\), ko-fi"],
    },
    "linus_(jarnqk)": {
        "character": ["linus_(jarnqk)"],
        "trigger": ["linus \\(jarnqk\\), cult of the lamb"],
    },
    "mikah_miller_(character)": {
        "character": ["mikah_miller_(character)"],
        "trigger": ["mikah miller \\(character\\), animal crossing"],
    },
    "x-38_(maddeku)": {
        "character": ["x-38_(maddeku)"],
        "trigger": ["x-38 \\(maddeku\\), tamagotchi"],
    },
    "jacob_(pablo)": {
        "character": ["jacob_(pablo)"],
        "trigger": ["jacob \\(pablo\\), patreon"],
    },
    "whyte_(daemon_lady)": {
        "character": ["whyte_(daemon_lady)"],
        "trigger": ["whyte \\(daemon lady\\), mythology"],
    },
    "choi_yujin": {
        "character": ["choi_yujin"],
        "trigger": ["choi yujin, the suicider rat"],
    },
    "reindeer_(petruz)": {
        "character": ["reindeer_(petruz)"],
        "trigger": ["reindeer \\(petruz\\), petruz \\(copyright\\)"],
    },
    "fursona_(birdpaw)": {
        "character": ["fursona_(birdpaw)"],
        "trigger": ["fursona \\(birdpaw\\), mythology"],
    },
    "juicy_(juicyghost)": {
        "character": ["juicy_(juicyghost)"],
        "trigger": ["juicy \\(juicyghost\\), arizona iced tea"],
    },
    "hermia_idril": {
        "character": ["hermia_idril"],
        "trigger": ["hermia idril, transisters"],
    },
    "smite_(character)": {
        "character": ["smite_(character)"],
        "trigger": ["smite \\(character\\), riot games"],
    },
    "purradise_meowscles": {
        "character": ["purradise_meowscles"],
        "trigger": ["purradise meowscles, fortnite"],
    },
    "kate_(father_of_the_pride)": {
        "character": ["kate_(father_of_the_pride)"],
        "trigger": ["kate \\(father of the pride\\), father of the pride"],
    },
    "caltsar": {"character": ["caltsar"], "trigger": ["caltsar, mythology"]},
    "tat_(klonoa)": {
        "character": ["tat_(klonoa)"],
        "trigger": ["tat \\(klonoa\\), bandai namco"],
    },
    "suika_ibuki": {"character": ["suika_ibuki"], "trigger": ["suika ibuki, touhou"]},
    "aardy": {"character": ["aardy"], "trigger": ["aardy, mythology"]},
    "lady_kluck": {"character": ["lady_kluck"], "trigger": ["lady kluck, disney"]},
    "taryn_crimson": {
        "character": ["taryn_crimson"],
        "trigger": ["taryn crimson, five nights at freddy's: security breach"],
    },
    "benji_(bng)": {
        "character": ["benji_(bng)"],
        "trigger": ["benji \\(bng\\), mythology"],
    },
    "deacon_chaos": {
        "character": ["deacon_chaos"],
        "trigger": ["deacon chaos, mythology"],
    },
    "snufkin": {"character": ["snufkin"], "trigger": ["snufkin, the moomins"]},
    "ishishi": {"character": ["ishishi"], "trigger": ["ishishi, kaiketsu zorori"]},
    "mandy_(tgaobam)": {
        "character": ["mandy_(tgaobam)"],
        "trigger": ["mandy \\(tgaobam\\), cartoon network"],
    },
    "jake_sully": {
        "character": ["jake_sully"],
        "trigger": ["jake sully, james cameron's avatar"],
    },
    "dino_(flintstones)": {
        "character": ["dino_(flintstones)"],
        "trigger": ["dino \\(flintstones\\), the flintstones"],
    },
    "copper_(animal_crossing)": {
        "character": ["copper_(animal_crossing)"],
        "trigger": ["copper \\(animal crossing\\), animal crossing"],
    },
    "gunther_hausmann": {
        "character": ["gunther_hausmann"],
        "trigger": ["gunther hausmann, good cheese"],
    },
    "amber_(scooby-doo)": {
        "character": ["amber_(scooby-doo)"],
        "trigger": ["amber \\(scooby-doo\\), scooby-doo \\(series\\)"],
    },
    "elmo": {"character": ["elmo"], "trigger": ["elmo, sesame street"]},
    "galacta_knight": {
        "character": ["galacta_knight"],
        "trigger": ["galacta knight, kirby \\(series\\)"],
    },
    "clubbon": {"character": ["clubbon"], "trigger": ["clubbon, mythology"]},
    "koba_(koba)": {
        "character": ["koba_(koba)"],
        "trigger": ["koba \\(koba\\), pokemon"],
    },
    "jack_(mass_effect)": {
        "character": ["jack_(mass_effect)"],
        "trigger": ["jack \\(mass effect\\), electronic arts"],
    },
    "general_grievous": {
        "character": ["general_grievous"],
        "trigger": ["general grievous, star wars"],
    },
    "sachel": {"character": ["sachel"], "trigger": ["sachel, halloween"]},
    "ziggy_zerda": {
        "character": ["ziggy_zerda"],
        "trigger": ["ziggy zerda, mythology"],
    },
    "noriko_takahashi": {
        "character": ["noriko_takahashi"],
        "trigger": ["noriko takahashi, milkjunkie"],
    },
    "sabrina_(housepets!)": {
        "character": ["sabrina_(housepets!)"],
        "trigger": ["sabrina \\(housepets!\\), housepets!"],
    },
    "avalondragon": {
        "character": ["avalondragon"],
        "trigger": ["avalondragon, mythology"],
    },
    "amy_pratt": {"character": ["amy_pratt"], "trigger": ["amy pratt, nintendo"]},
    "black_dragon_kalameet": {
        "character": ["black_dragon_kalameet"],
        "trigger": ["black dragon kalameet, fromsoftware"],
    },
    "hashimoto-chan": {
        "character": ["hashimoto-chan"],
        "trigger": ["hashimoto-chan, nintendo"],
    },
    "eve_cadrey": {"character": ["eve_cadrey"], "trigger": ["eve cadrey, raven wolf"]},
    "rai_(wyntersun)": {
        "character": ["rai_(wyntersun)"],
        "trigger": ["rai \\(wyntersun\\), mythology"],
    },
    "porunga": {"character": ["porunga"], "trigger": ["porunga, dragon ball"]},
    "rhoda": {"character": ["rhoda"], "trigger": ["rhoda, mayfield"]},
    "bentina_beakley": {
        "character": ["bentina_beakley"],
        "trigger": ["bentina beakley, disney"],
    },
    "vao_(coffeechicken)": {
        "character": ["vao_(coffeechicken)"],
        "trigger": ["vao \\(coffeechicken\\), don bluth"],
    },
    "kojote": {"character": ["kojote"], "trigger": ["kojote, bad dragon"]},
    "rorik_ironwill": {
        "character": ["rorik_ironwill"],
        "trigger": ["rorik ironwill, guild wars"],
    },
    "jaiy": {"character": ["jaiy"], "trigger": ["jaiy, limes guy"]},
    "breezie_the_hedgehog_(archie)": {
        "character": ["breezie_the_hedgehog_(archie)"],
        "trigger": [
            "breezie the hedgehog \\(archie\\), sonic the hedgehog \\(series\\)"
        ],
    },
    "wes_(pokemon)": {
        "character": ["wes_(pokemon)"],
        "trigger": ["wes \\(pokemon\\), pokemon colosseum"],
    },
    "kiba_(kiba32)": {
        "character": ["kiba_(kiba32)"],
        "trigger": ["kiba \\(kiba32\\), pokemon"],
    },
    "verde_the_snivy": {
        "character": ["verde_the_snivy"],
        "trigger": ["verde the snivy, pokemon"],
    },
    "beatriz_resont": {
        "character": ["beatriz_resont"],
        "trigger": ["beatriz resont, beatriz overseer"],
    },
    "ruby_lareme_(battler)": {
        "character": ["ruby_lareme_(battler)"],
        "trigger": ["ruby lareme \\(battler\\), cub con"],
    },
    "teraunce": {"character": ["teraunce"], "trigger": ["teraunce, mythology"]},
    "audrey_(woofyrainshadow)": {
        "character": ["audrey_(woofyrainshadow)"],
        "trigger": ["audrey \\(woofyrainshadow\\), subscribestar"],
    },
    "jetta_the_jolteon": {
        "character": ["jetta_the_jolteon"],
        "trigger": ["jetta the jolteon, pokemon"],
    },
    "victor_(brushfire)": {
        "character": ["victor_(brushfire)"],
        "trigger": ["victor \\(brushfire\\), mythology"],
    },
    "dashing_wanderer_ampharos": {
        "character": ["dashing_wanderer_ampharos"],
        "trigger": ["dashing wanderer ampharos, pokemon mystery dungeon"],
    },
    "ying": {"character": ["ying"], "trigger": ["ying, pokemon"]},
    "a0n_(a0nmaster)": {
        "character": ["a0n_(a0nmaster)"],
        "trigger": ["a0n \\(a0nmaster\\), a big comparison"],
    },
    "gardie_(otukimi)": {
        "character": ["gardie_(otukimi)"],
        "trigger": ["gardie \\(otukimi\\), mythology"],
    },
    "stardragon": {"character": ["stardragon"], "trigger": ["stardragon, mythology"]},
    "maly_paczek": {"character": ["maly_paczek"], "trigger": ["maly paczek, cuehors"]},
    "raven_darkfur": {
        "character": ["raven_darkfur"],
        "trigger": ["raven darkfur, mythology"],
    },
    "justin_(aaron)": {
        "character": ["justin_(aaron)"],
        "trigger": ["justin \\(aaron\\), mythology"],
    },
    "bombshell_(nitw)": {
        "character": ["bombshell_(nitw)"],
        "trigger": ["bombshell \\(nitw\\), night in the woods"],
    },
    "amali_(tloz)": {
        "character": ["amali_(tloz)"],
        "trigger": ["amali \\(tloz\\), breath of the wild"],
    },
    "storm_feather": {
        "character": ["storm_feather"],
        "trigger": ["storm feather, my little pony"],
    },
    "cornica_sonoma": {
        "character": ["cornica_sonoma"],
        "trigger": ["cornica sonoma, stranger things"],
    },
    "kayz_(snepkayz)": {
        "character": ["kayz_(snepkayz)"],
        "trigger": ["kayz \\(snepkayz\\), mythology"],
    },
    "ghost_(claralaine)": {
        "character": ["ghost_(claralaine)"],
        "trigger": ["ghost \\(claralaine\\), patreon"],
    },
    "unseenpanther": {
        "character": ["unseenpanther"],
        "trigger": ["unseenpanther, mythology"],
    },
    "cypherwolf": {
        "character": ["cypherwolf"],
        "trigger": ["cypherwolf, sony corporation"],
    },
    "takkun_(takkun7635)": {
        "character": ["takkun_(takkun7635)"],
        "trigger": ["takkun \\(takkun7635\\), brawlhalla"],
    },
    "ren_(remanedur)": {
        "character": ["ren_(remanedur)"],
        "trigger": ["ren \\(remanedur\\), mythology"],
    },
    "speed_(one_piece)": {
        "character": ["speed_(one_piece)"],
        "trigger": ["speed \\(one piece\\), one piece"],
    },
    "mewgle_(character)": {
        "character": ["mewgle_(character)"],
        "trigger": ["mewgle \\(character\\)"],
    },
    "brittany_(roushfan5)": {
        "character": ["brittany_(roushfan5)"],
        "trigger": ["brittany \\(roushfan5\\), christmas"],
    },
    "anthor": {"character": ["anthor"], "trigger": ["anthor, pokemon"]},
    "kurnak": {"character": ["kurnak"], "trigger": ["kurnak, greek mythology"]},
    "angelise_reiter": {
        "character": ["angelise_reiter"],
        "trigger": ["angelise reiter, square enix"],
    },
    "seiko_(chewycontroller)": {
        "character": ["seiko_(chewycontroller)"],
        "trigger": ["seiko \\(chewycontroller\\), christmas"],
    },
    "ivy_sundew": {"character": ["ivy_sundew"], "trigger": ["ivy sundew, disney"]},
    "cass_(simplifypm)": {
        "character": ["cass_(simplifypm)"],
        "trigger": ["cass \\(simplifypm\\), mythology"],
    },
    "sitri_(james_howard)": {
        "character": ["sitri_(james_howard)"],
        "trigger": ["sitri \\(james howard\\), subscribestar"],
    },
    "rouge_the_nun": {
        "character": ["rouge_the_nun"],
        "trigger": ["rouge the nun, sonic the hedgehog \\(series\\)"],
    },
    "els_(beastars)": {
        "character": ["els_(beastars)"],
        "trigger": ["els \\(beastars\\), beastars"],
    },
    "kura_(zoohomme)": {
        "character": ["kura_(zoohomme)"],
        "trigger": ["kura \\(zoohomme\\), zoohomme"],
    },
    "hexatoy": {"character": ["hexatoy"], "trigger": ["hexatoy, nintendo"]},
    "skipper(zoran)": {
        "character": ["skipper(zoran)"],
        "trigger": ["skipper\\(zoran\\), goodnites"],
    },
    "orion_(jacobjones14)": {
        "character": ["orion_(jacobjones14)"],
        "trigger": ["orion \\(jacobjones14\\), pokemon"],
    },
    "frogdor": {"character": ["frogdor"], "trigger": ["frogdor, nintendo"]},
    "paige_(snapshotstami)": {
        "character": ["paige_(snapshotstami)"],
        "trigger": ["paige \\(snapshotstami\\), amaverse"],
    },
    "prisma_(fr0gv0re)": {
        "character": ["prisma_(fr0gv0re)"],
        "trigger": ["prisma \\(fr0gv0re\\), minecraft"],
    },
    "lester_(risenpaw)": {
        "character": ["lester_(risenpaw)"],
        "trigger": ["lester \\(risenpaw\\), pokemon"],
    },
    "zara_(dalwart)": {
        "character": ["zara_(dalwart)"],
        "trigger": ["zara \\(dalwart\\), new year"],
    },
    "ryzz_(xeoniios)": {
        "character": ["ryzz_(xeoniios)"],
        "trigger": ["ryzz \\(xeoniios\\), mythology"],
    },
    "peachy_(marshmallow-ears)": {
        "character": ["peachy_(marshmallow-ears)"],
        "trigger": ["peachy \\(marshmallow-ears\\), pokemon"],
    },
    "bird_(petruz)": {
        "character": ["bird_(petruz)"],
        "trigger": ["bird \\(petruz\\), petruz \\(copyright\\)"],
    },
    "hermann_(knights_college)": {
        "character": ["hermann_(knights_college)"],
        "trigger": ["hermann \\(knights college\\), knights college"],
    },
    "gharn_(vju79)": {
        "character": ["gharn_(vju79)"],
        "trigger": ["gharn \\(vju79\\), mythology"],
    },
    "lia-lioness": {
        "character": ["lia-lioness"],
        "trigger": ["lia-lioness, cinco de mayo"],
    },
    "tabi_(fnf)": {
        "character": ["tabi_(fnf)"],
        "trigger": ["tabi \\(fnf\\), friday night funkin'"],
    },
    "chevy_dahl": {"character": ["chevy_dahl"], "trigger": ["chevy dahl, mythology"]},
    "soulsong_(celestial_wolf)": {
        "character": ["soulsong_(celestial_wolf)"],
        "trigger": ["soulsong \\(celestial wolf\\), pokemon"],
    },
    "mimi_(mr.smile)": {
        "character": ["mimi_(mr.smile)"],
        "trigger": ["mimi \\(mr.smile\\), pokemon"],
    },
    "vaporunny_(japeal)": {
        "character": ["vaporunny_(japeal)"],
        "trigger": ["vaporunny \\(japeal\\), pokemon"],
    },
    "horny_blue_bowlingball": {
        "character": ["horny_blue_bowlingball"],
        "trigger": ["horny blue bowlingball, wyer bowling \\(meme\\)"],
    },
    "apricot_(viroveteruscy)": {
        "character": ["apricot_(viroveteruscy)"],
        "trigger": ["apricot \\(viroveteruscy\\), warning cream filled"],
    },
    "blueberry_jam_(viroveteruscy)": {
        "character": ["blueberry_jam_(viroveteruscy)"],
        "trigger": ["blueberry jam \\(viroveteruscy\\), warning cream filled"],
    },
}
